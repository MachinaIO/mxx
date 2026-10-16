import Stage_preprocess_0_0_Part_0
import W_Stage_preprocess_0_0
import ExponentSampling

/-! Every run of the average-case relations of `Stage_preprocess_0_0_Part_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_preprocess_0_0_Part_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_3.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_3.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_3.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs19 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13.bridge hs19))), h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), (MxxExponentLut.gaussianSampleAt_sample (by decide) h17), h18⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_3.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_3 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_3 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_3
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_12_0, wit1.w_13_0, wit1.w_17_0, wit1.w_18_0, wit1.w_21_0, wit1.w_30_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_3.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_4.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_4.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_4.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_4.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_4 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_4 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_4
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_4.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_5.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_5.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_5.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs19 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13.bridge hs19))), h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), (MxxExponentLut.gaussianSampleAt_sample (by decide) h17), h18⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_5.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_5 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_5 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_5
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_12_0, wit1.w_13_0, wit1.w_17_0, wit1.w_18_0, wit1.w_21_0, wit1.w_30_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_5.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_6.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_6.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_6.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs19 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13.bridge hs19))), h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), (MxxExponentLut.gaussianSampleAt_sample (by decide) h17), h18⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_6.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_6 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_6 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_6
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_12_0, wit1.w_13_0, wit1.w_17_0, wit1.w_18_0, wit1.w_21_0, wit1.w_30_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_6.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_7.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_7.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_7.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_7.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_7 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_7 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_7
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_7.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_8.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_8.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_8.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_8.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_8 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_8 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_8
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_8.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_9.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_9.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_9.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_9.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_9 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_9 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_9
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_9.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_10.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_10.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_10.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_10.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_10 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_10 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_10
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_10.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_11.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_11.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_11.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_11.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_11 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_11 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_11
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_11.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_12.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_12.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_12.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_12.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_12 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_12 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_12
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_12.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_13.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_13 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_13 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_13
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_14.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_14.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_14.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_14.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_14 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_14 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_14
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_14.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_15.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_15.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_15.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_15.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_15 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_15 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_15
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_15.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_16_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_16_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_16_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_16_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_16_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_16_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_16_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_16_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_16.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_16.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_16.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_16_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_16_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_16.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_16 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_16 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_16
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_16.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_17_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_17_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_17_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_17_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_17_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_17_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_17_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_17_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_17.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_17.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_17.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_17_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_17_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_17.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_17 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_17 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_17
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_17.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_18_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_18_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_18_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_18_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_18_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_18_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_18_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_18_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_18.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_18.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_18.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_18_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_18_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_18.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_18 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_18 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_18
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_18.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_19_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_19_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_19_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_19_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_19_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_19_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_19_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_19_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_19.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_19.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_19.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_19_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_19_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_19.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_19 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_19 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_19
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_19.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_20_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_20_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_20_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_20_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_20_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_20_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_20_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_20_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_20_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_20_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_20.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_20 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_20 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_20
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_21_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_21_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_21_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_21_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_21_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_21_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_21_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_21_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_21.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_21.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_21.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_21_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_21_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_21.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_21 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_21 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_21
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_21.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_22_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_22_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_22_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_22_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_22_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_22_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_22_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_22_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_22.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_22.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_22.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_22_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_22_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_22.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_22 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_22 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_22
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_22.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_23_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_23_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_23_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_23_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_23_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_23_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_23_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_23_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_23.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_23.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_23.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_23_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_23_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_23.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_23 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_23 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_23
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_23.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_24_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_24_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_24_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_24_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_24_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_24_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_24_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_24_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_24_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_24_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_24.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_24 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_24 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_24
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_25_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_25_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_25_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_25_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_25_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_25_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_25_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_25_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_25.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_25.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_25.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_25_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_25_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_25.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_25 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_25 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_25
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_25.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_26_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_26_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_26_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_26_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_26_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_26_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_26_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_26_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_26.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_26.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_26.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_26_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_26_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_26.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_26 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_26 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_26
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_26.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_27_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_27_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_27_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_27_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_27_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_27_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_27_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_27_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_27.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_27.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_27.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_27_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_27_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_27.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_27 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_27 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_27
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_27.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_28_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_28_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_28_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_28_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_28_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_28_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_28_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_28_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_28.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_28.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_28.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_28_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_28_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_28.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_28 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_28 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_28
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_28.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_29_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_29_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_29_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_29_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_29_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_29_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_29_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_29_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_29.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_29.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_29.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_29_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_29_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_29.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_29 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_29 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_29
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_29.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_30_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_30_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_30_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_30_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_30_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_30_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_30_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_30_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_30.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_30.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_30.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_30_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_30_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_30.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_30 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_30 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_30
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_30.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_31_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_31_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_31_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_31_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_31_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_31_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_31_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_31_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_31.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_31.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_31.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_31_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_31_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_31.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_31 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_31 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_31
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_31.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_32_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_32_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_32_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_32_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_32_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_32_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_32_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_32_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_32.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_32.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_32.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_32_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_32_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_32.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_32 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_32 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_32
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_32.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_33_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_33_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_33_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_33_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_33_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_33_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_33_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_33_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_33.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_33.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_33.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_33_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_33_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_33.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_33 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_33 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_33
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_33.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_34_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_34_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_34_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_34_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_34_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_34_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_34_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_34_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_34.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_34.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_34.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_34_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_34_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_34.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_34 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_34 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_34
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_34.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_35_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_35_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_35_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_35_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_35_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_35_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_35_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_35_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_35_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_35_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_35.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_35 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_35 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_35
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_36_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_36_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_36_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_36_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_36_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_36_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_36_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_36_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_36.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_36.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_36.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_36_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_36_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_36.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_36 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_36 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_36
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_36.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_37_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_37_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_37_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_37_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_37_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_37_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_37_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_37_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_37.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_37.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_37.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_37_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_37_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_37.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_37 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_37 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_37
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_37.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_38_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_38_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_38_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_38_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_38_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_38_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_38_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_38_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_38.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_38.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_38.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_38_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_38_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_38.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_38 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_38 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_38
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_38.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_39_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_39_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_39_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_39_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_39_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_39_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_39_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_39_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_39_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_39_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_39.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_39 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_39 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_39
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_40_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_40_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_40_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_40_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_40_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_40_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_40_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_40_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_40.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_40.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_40.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_40_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_40_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_40.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_40 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_40 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_40
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_40.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_41_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_41_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_41_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_41_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_41_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_41_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_41_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_41_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_41.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_41.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_41.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_41_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_41_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_41.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_41 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_41 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_41
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_41.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_42_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_42_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_42_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_42_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_42.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_42.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_42.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_42_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_42.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_42 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_42 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_42
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_42.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_43_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_43_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_43_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_43_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_43_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_43_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_43_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_43_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_43.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_43.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_43.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_43_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_43_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_43.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_43 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_43 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_43
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_43.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_44_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_44_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_44_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_44_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_44_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_44_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_44_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_44_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_44.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_44.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_44.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_44_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_44_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_44.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_44 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_44 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_44
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_44.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_45_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_45_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_45_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_45_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_45_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_45_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_45_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_45_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_45.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_45.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_45.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_45_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_45_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_45.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_45 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_45 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_45
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_45.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_46_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_46_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_46_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_46_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_46_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_46_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_46_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_46_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_46.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_46.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_46.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_46_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_46_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_46.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_46 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_46 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_46
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_46.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_47_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_47_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_47_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_47_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_47_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_47_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_47_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_47_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_47.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_47.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_47.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_47_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_47_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_47.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_47 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_47 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_47
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_47.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_48_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_48_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_48_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_48_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_48_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_48_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_48_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_48_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_48.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_48.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_48.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_48_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_48_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_48.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_48 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_48 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_48
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_48.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_49_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_49_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_49_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_49_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_49_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_49_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_49_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_49_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_49.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_49.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_49.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_49_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_49_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_49.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_49 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_49 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_49
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_49.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_50_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_50_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_50_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_50_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_50_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_50_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_50_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_50_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_50.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_50.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_50.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_50_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_50_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_50.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_50 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_50 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_50
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_50.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_51_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_51_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_51_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_51_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_51_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_51_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_51_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_51_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_51.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_51.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_51.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_51_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_51_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_51.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_51 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_51 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_51
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_51.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_52_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_52_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_52_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_52_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_52_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_52_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_52_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_52_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_52.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_52.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_52.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_52_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_52_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_52.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_52 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_52 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_52
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_52.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_53_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_53_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_53_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_53_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_53_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_53_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_53_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_53_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_53.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_53.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_53.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_53_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_53_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_53.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_53 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_53 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_53
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_53.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_54_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_54_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_54_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_54_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_54_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_54_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_54_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_54_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_54.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_54.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_54.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_54_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_54_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_54.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_54 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_54 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_54
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_54.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_55_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_55_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_55_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_55_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_55_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_55_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_55_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_55_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_55.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_55.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_55.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_55_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_55_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_55.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_55 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_55 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_55
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_55.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_56_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_56_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_56_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_56_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_56_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_56_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_56_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_56_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_56.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_56.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_56.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_56_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_56_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_56.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_56 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_56 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_56
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_56.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_57_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_57_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_57_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_57_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_57_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_57_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_57_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_57_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_57.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_57.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_57.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_57_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_57_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_57.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_57 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_57 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_57
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_57.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_58_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_58_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_58_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_58_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_58_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_58_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_58_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_58_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_58.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_58.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_58.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_58_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_58_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_58.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_58 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_58 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_58
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_58.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_59_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_59_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_59_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_59_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_59_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_59_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_59_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_59_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_59.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_59.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_59.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_59_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_59_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_59.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_59 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_59 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_59
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_59.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_60_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_60_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_60_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_60_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_60_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_60_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_60_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_60_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_60.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_60.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_60.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_60_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_60_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_60.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_60 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_60 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_60
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_60.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_61_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_61_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_61_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_61_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_61_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_61_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_61_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_61_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_61.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_61.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_61.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_61_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_61_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_61.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_61 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_61 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_61
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_61.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_62_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_62_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_62_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_62_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_62_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_62_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_62_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_62_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_62.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_62.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_62.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_62_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_62_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_62.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_62 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_62 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_62
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_62.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_63_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_63_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_63_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_63_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_63_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_63_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_63_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_63_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_63.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_63.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_63.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_63_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_63_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_63.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_63 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_63 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_63
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_63.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_64_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_64_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_64_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_64_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_64_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_64_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_64_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_64_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_64.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_64.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_64.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_64_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_64_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_64.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_64 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_64 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_64
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_64.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_65_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_65_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_65_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_65_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_65_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_65_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_65_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_65_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_65.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_65.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_65.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_65_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_65_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_65.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_65 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_65 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_65
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_65.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_66_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_66_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_66_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_66_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_66_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_66_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_66_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_66_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_66.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_66.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_66.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_66_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_66_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_66.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_66 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_66 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_66
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_66.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_67_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_67_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_67_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_67_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_67_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_67_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_67_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_67_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_67.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_67.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_67.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_67_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_67_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_67.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_67 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_67 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_67
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_67.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_68_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_68_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_68_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_68_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_68_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_68_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_68_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_68_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_68.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_68.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_68.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_68_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_68_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_68.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_68 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_68 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_68
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_68.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_69_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_69_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_69_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_69_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_69_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_69_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_69_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_69_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_69.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_69.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_69.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_69_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_69_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_69.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_69 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_69 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_69
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_69.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_70_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_70_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_70_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_70_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_70_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_70_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_70_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_70_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_70.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_70.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_70.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_70_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_70_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_70.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_70 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_70 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_70
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_70.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_71_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_71_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_71_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_71_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_71_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_71_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_71_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_71_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_71.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_71.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_71.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_71_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_71_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_71.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_71 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_71 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_71
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_71.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_72_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_72_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_72_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_72_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_72_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_72_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_72_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_72_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_72.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_72.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_72.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_72_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_72_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_72.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_72 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_72 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_72
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_72.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_73_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_73_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_73_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_73_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_73_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_73_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_73_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_73_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_73.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_73.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_73.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_73_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_73_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_73.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_73 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_73 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_73
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_73.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_74_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_74_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_74_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_74_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_74_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_74_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_74_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_74_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_74.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_74.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_74.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_74_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_74_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_74.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_74 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_74 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_74
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_74.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_75_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_75_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_75_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_75_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_75_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_75_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_75_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_75_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_75.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_75.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_75.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_75_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_75_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_75.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_75 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_75 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_75
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_75.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_76_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_76_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_76_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_76_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_76_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_76_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_76_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_76_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_76.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_76.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_76.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_76_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_76_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_76.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_76 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_76 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_76
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_76.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_77_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_77_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_77_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_77_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_77_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_77_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_77_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_77_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_77.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_77.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_77.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_77_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_77_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_77.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_77 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_77 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_77
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_77.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_78_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_78_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_78_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_78_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_78_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_78_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_78_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_78_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_78.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_78.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_78.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_78_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_78_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_78.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_78 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_78 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_78
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_78.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_79_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_79_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_79_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_79_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_79_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_79_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_79_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_79_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_79.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_79.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_79.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_79_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_79_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_79.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_79 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_79 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_79
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_79.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_80.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_80.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_80.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_80.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_80 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_80 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_80
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_80.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_81_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_81_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_81_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_81_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_81_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_81_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_81_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_81_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_81.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_81.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_81.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_81_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_81_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_81.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_81 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_81 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_81
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_81.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_82_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_82_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_82_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_82_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_82_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_82_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_82_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_82_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_82.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_82.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_82.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_82_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_82_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_82.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_82 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_82 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_82
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_82.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_83_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_83_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_83_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_83_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_83_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_83_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_83_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_83_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_83.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_83.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_83.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_83_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_83_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_83.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_83 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_83 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_83
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_83.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_84.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_84.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_84.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_84.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_84 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_84 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_84
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_84.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_85_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_85_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_85_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_85_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_85.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_85.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_85.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_85_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_85.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_85 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_85 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_85
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_85.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_86_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_86_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_86_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_86_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_86_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_86_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_86_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_86_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_86.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_86.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_86.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_86_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_86_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_86.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_86 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_86 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_86
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_86.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_87_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_87_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_87_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_87_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_87_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_87_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_87_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_87_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_87.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_87.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_87.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_87_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_87_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_87.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_87 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_87 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_87
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_87.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_88_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_88_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_88_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_88_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_88_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_88_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_88_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_88_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_88.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_88.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_88.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_88_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_88_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_88.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_88 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_88 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_88
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_88.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_89_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_89_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_89_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_89_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_89_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_89_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_89_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_89_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_89.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_89.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_89.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_89_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_89_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_89.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_89 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_89 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_89
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_89.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_90_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_90_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_90_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_90_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_90_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_90_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_90_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_90_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_90.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_90.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_90.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_90_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_90_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_90.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_90 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_90 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_90
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_90.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_91_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_91_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_91_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_91_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_91_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_91_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_91_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_91_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_91.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_91.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_91.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_91_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_91_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_91.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_91 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_91 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_91
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_91.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_92_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_92_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_92_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_92_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_92_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_92_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_92_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_92_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_92.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_92.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_92.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_92_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_92_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_92.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_92 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_92 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_92
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_92.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_93_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_93_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_93_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_93_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_93_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_93_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_93_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_93_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_93.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_93.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_93.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_93_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_93_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_93.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_93 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_93 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_93
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_93.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_94_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_94_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_94_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_94_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_94_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_94_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_94_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_94_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_94.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_94.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_94.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_94_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_94_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_94.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_94 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_94 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_94
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_94.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_95_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_95_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_95_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_95_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_95_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_95_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_95_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_95_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_95.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_95.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_95.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_95_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_95_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_95.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_95 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_95 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_95
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_95.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_96_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_96_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_96_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_96_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_96_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_96_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_96_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_96_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_96.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_96.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_96.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_96_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_96_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_96.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_96 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_96 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_96
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_96.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_97_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_97_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_97_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_97_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_97_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_97_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_97_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_97_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_97.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_97.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_97.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_97_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_97_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_97.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_97 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_97 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_97
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_97.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_98_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_98_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_98_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_98_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_98_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_98_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_98_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_98_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_98.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_98.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_98.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_98_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_98_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_98.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_98 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_98 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_98
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_98.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_99_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_99_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_99_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_99_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_99_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_99_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_99_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_99_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_99.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_99.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_99.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_99_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_99_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_99.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_99 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_99 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_99
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_99.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_100_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_100_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_100_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_100_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_100_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_100_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_100_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_100_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_100.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_100.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_100.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_100_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_100_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_100.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_100 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_100 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_100
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_100.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_101_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_101_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_101_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_101_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_101_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_101_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_101_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_101_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_101.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_101.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_101.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_101_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_101_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_101.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_101 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_101 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_101
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_101.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_102_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_102_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_102_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_102_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_102_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_102_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_102_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_102_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_102.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_102.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_102.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_102_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_102_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_102.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_102 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_102 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_102
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_102.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_103_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_103_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_103_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_103_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_103_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_103_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_103_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_103_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_103.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_103.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_103.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_103_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_103_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_103.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_103 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_103 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_103
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_103.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_104_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_104_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_104_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_104_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_104_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_104_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_104_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_104_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_104.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_104.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_104.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_104_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_104_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_104.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_104 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_104 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_104
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_104.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_105_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_105_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_105_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_105_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_105_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_105_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_105_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_105_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_105.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_105.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_105.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_105_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_105_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_105.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_105 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_105 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_105
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_105.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_106_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_106_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_106_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_106_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_106_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_106_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_106_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_106_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_106.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_106.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_106.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_106_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_106_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_106.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_106 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_106 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_106
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_106.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_107_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_107_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_107_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_107_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_107_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_107_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_107_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_107_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_107.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_107.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_107.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_107_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_107_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_107.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_107 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_107 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_107
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_107.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_108_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_108_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_108_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_108_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_108_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_108_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_108_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_108_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_108.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_108.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_108.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_108_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_108_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_108.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_108 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_108 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_108
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_108.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_109_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_109_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_109_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_109_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_109_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_109_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_109_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_109_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_109.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_109.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_109.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_109_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_109_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_109.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_109 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_109 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_109
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_109.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_110_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_110_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_110_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_110_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_110_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_110_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_110_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_110_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_110.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_110.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_110.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_110_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_110_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_110.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_110 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_110 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_110
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_110.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_111_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_111_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_111_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_111_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_111_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_111_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_111_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_111_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_111.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_111.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_111.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_111_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_111_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_111.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_111 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_111 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_111
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_111.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_112_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_112_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_112_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_112_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_112_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_112_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_112_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_112_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_112.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_112.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_112.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_112_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_112_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_112.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_112 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_112 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_112
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_112.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_113_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_113_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_113_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_113_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_113_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_113_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_113_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_113_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_113.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_113.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_113.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_113_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_113_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_113.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_113 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_113 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_113
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_113.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_114_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_114_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_114_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_114_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_114_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_114_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_114_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_114_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_114.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_114.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_114.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_114_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_114_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_114.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_114 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_114 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_114
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_114.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_115_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_115_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_115_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_115_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_115_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_115_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_115_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_115_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_115.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_115.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_115.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_115_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_115_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_115.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_115 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_115 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_115
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_115.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_116_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_116_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_116_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_116_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_116_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_116_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_116_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_116_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_116.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_116.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_116.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_116_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_116_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_116.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_116 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_116 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_116
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_116.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_117.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_117.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_117.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_117.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_117 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_117 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_117
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_117.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_118_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_118_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_118_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_118_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_118_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_118_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_118_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_118_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_118.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_118.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_118.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_118_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_118_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_118.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_118 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_118 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_118
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_118.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_119_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_119_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_119_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_119_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_119_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_119_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_119_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_119_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_119.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_119.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_119.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_119_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_119_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_119.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_119 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_119 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_119
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_119.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_120_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_120_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_120_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_120_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_120_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_120_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_120_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_120_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_120.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_120.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_120.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_120_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_120_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_120.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_120 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_120 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_120
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_120.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_121.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_121.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_121.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_121.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_121 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_121 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_121
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_121.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_122_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_122_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_122_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_122_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_122.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_122.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_122.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_122_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_122.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_122 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_122 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_122
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_122.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_123_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_123_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_123_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_123_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_123.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_123.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_123.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_123_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_123.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_123 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_123 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_123
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_123.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_124_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_124_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_124_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_124_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_124_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_124_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_124_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_124_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_124.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_124.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_124.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_124_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_124_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_124.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_124 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_124 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_124
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_124.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_125_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_125_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_125_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_125_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_125_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_125_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_125_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_125_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_125.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_125.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_125.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_125_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_125_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_125.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_125 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_125 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_125
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_125.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_126_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_126_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_126_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_126_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_126_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_126_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_126_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_126_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_126.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_126.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_126.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_126_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_126_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_126.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_126 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_126 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_126
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_126.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_127_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_127_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_127_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_127_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_127_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_127_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_127_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_127_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_127.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_127.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_127.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_127_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_127_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_127.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_127 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_127 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_127
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_127.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_128_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_128_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_128_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_128_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_128_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_128_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_128_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_128_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_128.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_128.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_128.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_128_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_128_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_128.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_128 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_128 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_128
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_128.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_129_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_129_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_129_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_129_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_129_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_129_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_129_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_129_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_129.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_129.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_129.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_129_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_129_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_129.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_129 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_129 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_129
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_129.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_130_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_130_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_130_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_130_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_130_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_130_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_130_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_130_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_130.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_130.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_130.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_130_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_130_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_130.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_130 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_130 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_130
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_130.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_131_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_131_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_131_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_131_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_131_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_131_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_131_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_131_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_131.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_131.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_131.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_131_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_131_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_131.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_131 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_131 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_131
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_131.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_132_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_132_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_132_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_132_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_132_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_132_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_132_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_132_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_132.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_132.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_132.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_132_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_132_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_132.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_132 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_132 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_132
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_132.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_133_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_133_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_133_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_133_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_133_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_133_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_133_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_133_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_133.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_133.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_133.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_133_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_133_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_133.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_133 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_133 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_133
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_133.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_134_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_134_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_134_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_134_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_134_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_134_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_134_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_134_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_134.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_134.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_134.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_134_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_134_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_134.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_134 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_134 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_134
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_134.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_135_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_135_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_135_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_135_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_135_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_135_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_135_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_135_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_135.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_135.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_135.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_135_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_135_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_135.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_135 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_135 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_135
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_135.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_136_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_136_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_136_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_136_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_136_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_136_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_136_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_136_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_136.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_136.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_136.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_136_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_136_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_136.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_136 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_136 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_136
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_136.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_137_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_137_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_137_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_137_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_137_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_137_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_137_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_137_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_137.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_137.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_137.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_137_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_137_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_137.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_137 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_137 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_137
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_137.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_138_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_138_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_138_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_138_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_138_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_138_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_138_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_138_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_138.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_138.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_138.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_138_10.bridge (h11 i))), h12, (Stage_preprocess_0_0_Part_0.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_138_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_138.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_138 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_138 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_138
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_138.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_159.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_4_0 w_5_0 w_16_0 w_17_0 w_24_0 w_25_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_159.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_4_0 w_5_0 w_16_0 w_17_0 w_24_0 w_25_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_159.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_4_0 w_5_0 w_16_0 w_17_0 w_24_0 w_25_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h; exact ⟨h1, h2, h3, h4, (MxxExponentLut.gaussianSampleAt_sample (by decide) h5), (MxxExponentLut.gaussianSampleAt_sample (by decide) h6), h7⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_159.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_159 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_159 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_159
  exact (by obtain ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_16_0, w_24_0, h1⟩ := h; exact ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_16_0, w_24_0, (Stage_preprocess_0_0.parallel_generatedRoot_159.constraints_0.bridge h1)⟩)


theorem Stage_preprocess_0_0.sequential_generatedRoot_192.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_192 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_192 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_192
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_234.constraints_2.bridge {backend tape path w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_234.constraints_2 backend tape path w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_234.constraints_2 backend w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_234.constraints_1.bridge {backend hashModel tape path i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_234.constraints_1 backend hashModel tape path i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_234.constraints_1 backend hashModel i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, (MxxExponentLut.gaussianSampleAt_sample (by decide) h22), h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_234.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_234 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_234 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_234
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_19_0, w_20_0, w_21_0, w_24_0, w_27_0, w_30_0, w_31_0, w_32_0, w_34_0, w_35_0, w_46_0, w_48_0, w_49_0, w_52_0, w_53_0, w_54_0, w_55_0, w_63_0, w_65_0, w_66_0, w_69_0, w_70_0, w_71_0, h1, h2, h3⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_19_0, w_20_0, w_21_0, w_24_0, w_27_0, w_30_0, w_31_0, w_32_0, w_34_0, w_35_0, w_46_0, w_48_0, w_49_0, w_52_0, w_53_0, w_54_0, w_55_0, w_63_0, w_65_0, w_66_0, w_69_0, w_70_0, w_71_0, h1, (Stage_preprocess_0_0.parallel_generatedRoot_234.constraints_1.bridge h2), (Stage_preprocess_0_0.parallel_generatedRoot_234.constraints_2.bridge h3)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_235_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_235_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_235_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_235_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_235_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_235_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_235_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_235_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_235.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_235.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_235.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_235_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_235_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_235.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_235 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_235 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_235
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_235.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_238.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_238.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_238.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_238.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_238 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_238 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_238
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_238.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_239_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_239_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_239_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_239_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_239_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_239_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_239_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_239_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_239.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_239.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_239.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_239_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_239_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_239.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_239 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_239 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_239
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_239.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_242.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_242.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_242.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_242.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_242 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_242 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_242
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_242.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_243_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_243_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_243_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_243_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_243_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_243_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_243_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_243_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_243.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_243.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_243.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_243_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_243_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_243.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_243 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_243 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_243
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_243.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_246.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_246.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_246.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_246.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_246 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_246 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_246
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_246.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_247_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_247_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_247_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_247_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_247_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_247_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_247_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_247_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_247.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_247.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_247.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_247_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_247_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_247.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_247 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_247 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_247
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_247.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_250.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_250.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_250.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_250.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_250 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_250 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_250
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_250.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_251_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_251_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_251_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_251_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_251_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_251_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_251_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_251_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_251.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_251.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_251.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_251_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_251_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_251.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_251 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_251 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_251
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_251.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_254.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_254.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_254.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_254.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_254 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_254 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_254
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_254.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_255_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_255_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_255_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_255_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_255_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_255_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_255_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_255_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_255.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_255.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_255.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_255_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_255_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_255.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_255 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_255 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_255
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_255.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_258.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_258.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_258.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_258.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_258 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_258 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_258
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_258.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_259_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_259_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_259_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_259_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_259_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_259_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_259_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_259_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_259.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_259.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_259.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_259_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_259_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_259.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_259 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_259 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_259
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_259.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_262.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_262.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_262.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_262.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_262 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_262 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_262
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_262.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_263_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_263_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_263_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_263_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_263_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_263_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_263_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_263_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_263.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_263.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_263.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_263_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_263_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_263.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_263 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_263 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_263
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_263.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_266.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_266.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_266.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_266.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_266 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_266 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_266
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_266.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_267_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_267_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_267_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_267_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_267_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_267_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_267_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_267_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_267.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_267.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_267.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_267_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_267_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_267.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_267 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_267 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_267
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_267.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_270.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_270.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_270.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_270.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_270 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_270 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_270
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_270.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_271_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_271_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_271_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_271_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_271_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_271_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_271_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_271_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_271.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_271.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_271.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_271_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_271_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_271.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_271 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_271 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_271
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_271.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_274.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_274.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_274.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_274.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_274 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_274 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_274
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_274.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_275_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_275_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_275_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_275_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_275_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_275_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_275_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_275_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_275.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_275.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_275.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_275_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_275_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_275.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_275 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_275 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_275
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_275.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_278.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_278.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_278.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_278.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_278 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_278 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_278
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_278.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_279_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_279_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_279_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_279_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_279_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_279_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_279_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_279_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_279.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_279.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_279.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_279_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_279_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_279.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_279 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_279 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_279
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_279.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_282.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_282.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_282.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_282.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_282 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_282 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_282
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_282.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_283_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_283_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_283_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_283_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_283_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_283_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_283_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_283_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_283.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_283.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_283.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_283_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_283_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_283.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_283 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_283 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_283
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_283.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_286.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_286.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_286.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_286.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_286 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_286 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_286
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_286.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_287_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_287_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_287_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_287_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_287_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_287_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_287_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_287_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_287.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_287.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_287.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_287_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_287_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_287.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_287 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_287 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_287
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_287.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_290.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_290.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_290.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_290.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_290 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_290 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_290
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_290.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_291_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_291_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_291_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_291_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_291_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_291_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_291_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_291_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_291.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_291.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_291.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_291_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_291_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_291.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_291 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_291 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_291
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_291.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_294.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_294.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_294.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_294.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_294 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_294 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_294
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_294.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_295_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_295_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_295_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_295_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_295_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_295_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_295_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_295_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_295.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_295.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_295.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_295_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_295_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_295.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_295 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_295 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_295
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_295.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_298.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_298.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_298.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_298.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_298 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_298 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_298
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_298.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_299_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_299_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_299_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_299_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_299_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_299_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_299_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_299_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_299.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_299.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_299.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_299_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_299_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_299.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_299 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_299 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_299
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_299.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_302.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_302.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_302.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_302.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_302 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_302 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_302
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_302.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_303_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_303_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_303_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_303_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_303_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_303_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_303_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_303_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_303.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_303.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_303.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_303_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_303_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_303.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_303 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_303 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_303
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_303.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_306.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_306.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_306.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_306.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_306 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_306 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_306
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_306.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_307_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_307_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_307_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_307_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_307_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_307_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_307_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_307_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_307.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_307.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_307.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_307_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_307_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_307.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_307 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_307 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_307
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_307.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_310.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_310.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_310.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_310.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_310 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_310 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_310
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_310.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_311_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_311_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_311_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_311_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_311_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_311_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_311_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_311_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_311.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_311.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_311.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_311_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_311_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_311.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_311 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_311 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_311
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_311.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_314.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_314.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_314.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_314.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_314 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_314 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_314
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_314.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_315_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_315_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_315_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_315_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_315_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_315_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_315_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_315_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_315.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_315.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_315.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_315_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_315_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_315.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_315 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_315 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_315
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_315.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_318.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_318.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_318.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_318.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_318 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_318 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_318
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_318.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_319_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_319_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_319_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_319_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_319_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_319_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_319_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_319_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_319.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_319.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_319.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_319_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_319_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_319.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_319 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_319 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_319
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_319.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_322.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_322.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_322.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_322.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_322 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_322 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_322
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_322.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_323_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_323_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_323_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_323_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_323.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_323.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_323.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_323_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_323.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_323 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_323 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_323
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_323.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_326.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_326.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_326.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_326.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_326 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_326 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_326
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_326.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_327_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_327_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_327_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_327_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_327_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_327_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_327_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_327_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_327.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_327.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_327.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_327_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_327_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_327.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_327 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_327 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_327
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_327.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_330.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_330.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_330.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_330.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_330 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_330 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_330
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_330.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_331_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_331_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_331_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_331_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_331_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_331_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_331_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_331_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_331.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_331.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_331.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_331_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_331_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_331.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_331 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_331 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_331
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_331.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_334.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_334.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_334.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_334.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_334 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_334 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_334
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_334.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_335_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_335_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_335_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_335_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_335_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_335_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_335_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_335_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_335.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_335.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_335.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_335_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_335_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_335.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_335 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_335 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_335
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_335.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_338.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_338.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_338.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_338.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_338 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_338 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_338
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_338.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_339_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_339_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_339_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_339_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_339_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_339_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_339_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_339_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_339.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_339.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_339.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_339_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_339_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_339.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_339 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_339 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_339
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_339.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_342.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_342.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_342.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_342.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_342 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_342 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_342
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_342.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_343_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_343_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_343_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_343_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_343_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_343_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_343_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_343_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_343.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_343.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_343.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_343_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_343_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_343.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_343 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_343 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_343
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_343.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_346.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_346.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_346.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_346.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_346 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_346 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_346
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_346.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_347_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_347_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_347_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_347_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_347_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_347_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_347_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_347_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_347.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_347.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_347.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_347_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_347_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_347.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_347 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_347 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_347
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_347.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_350.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_350.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_350.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_350.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_350 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_350 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_350
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_350.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_351_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_351_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_351_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_351_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_351_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_351_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_351_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_351_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_351.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_351.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_351.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_351_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_351_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_351.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_351 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_351 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_351
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_351.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_354.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_354.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_354.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_354.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_354 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_354 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_354
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_354.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_355_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_355_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_355_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_355_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_355_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_355_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_355_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_355_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_355.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_355.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_355.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_355_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_355_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_355.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_355 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_355 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_355
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_355.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_358.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_358.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_358.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_358.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_358 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_358 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_358
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_358.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_359_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_359_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_359_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_359_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_359_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_359_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_359_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_359_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_359.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_359.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_359.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_359_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_359_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_359.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_359 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_359 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_359
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_359.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_362.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_362.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_362.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_362.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_362 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_362 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_362
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_362.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_363_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_363_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_363_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_363_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_363_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_363_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_363_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_363_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_363.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_363.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_363.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_363_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_363_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_363.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_363 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_363 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_363
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_363.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_366.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_366.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_366.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_366.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_366 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_366 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_366
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_366.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_367_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_367_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_367_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_367_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_367_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_367_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_367_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_367_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_367.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_367.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_367.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_367_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_367_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_367.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_367 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_367 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_367
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_367.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_368_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_368_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_368_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_368_10
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_368.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_368.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_368.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_0_0_Part_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_368_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_368.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_368 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_368 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_368
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_368.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_369.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_369 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_369 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_369
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_382.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_382 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_382 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_382
  exact h


theorem Stage_preprocess_0_0.scope_965.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.scope_965 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.scope_965 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.scope_965
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_390.constraints_0.bridge {backend hashModel tape path params params' i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_390.constraints_0 backend hashModel tape path params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0) : W_Stage_preprocess_0_0.sequential_generatedRoot_390.constraints_0 backend hashModel params' i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (Stage_preprocess_0_0_Part_0.iterRuns_mono h28 (fun _ _ _ hs33 ↦ (Stage_preprocess_0_0.scope_965.bridge hs33))), h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.sequential_generatedRoot_390.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_390 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_390 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_390
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_6_0, wit1.w_9_0, wit1.w_10_0, wit1.w_13_0, wit1.w_14_0, wit1.w_17_0, wit1.w_18_0, wit1.w_20_0, wit1.w_20_1, wit1.w_20_2, wit1.w_20_3, wit1.w_21_0, wit1.w_25_0, wit1.w_29_0, wit1.w_33_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_preprocess_0_0.sequential_generatedRoot_390.constraints_0.bridge h3), h4⟩)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_431.constraints_2.bridge {backend tape path w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_431.constraints_2 backend tape path w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_431.constraints_2 backend w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_431.constraints_1.bridge {backend hashModel tape path i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_431.constraints_1 backend hashModel tape path i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_431.constraints_1 backend hashModel i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, (MxxExponentLut.gaussianSampleAt_sample (by decide) h22), h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_431.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_431 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_431 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_431
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_19_0, w_20_0, w_21_0, w_24_0, w_27_0, w_30_0, w_31_0, w_32_0, w_34_0, w_35_0, w_46_0, w_48_0, w_49_0, w_52_0, w_53_0, w_54_0, w_55_0, w_63_0, w_65_0, w_66_0, w_69_0, w_70_0, w_71_0, h1, h2, h3⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_19_0, w_20_0, w_21_0, w_24_0, w_27_0, w_30_0, w_31_0, w_32_0, w_34_0, w_35_0, w_46_0, w_48_0, w_49_0, w_52_0, w_53_0, w_54_0, w_55_0, w_63_0, w_65_0, w_66_0, w_69_0, w_70_0, w_71_0, h1, (Stage_preprocess_0_0.parallel_generatedRoot_431.constraints_1.bridge h2), (Stage_preprocess_0_0.parallel_generatedRoot_431.constraints_2.bridge h3)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_432_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_432_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_432_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_432_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_432.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_432.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_432.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_432_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_432.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_432 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_432 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_432
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_432.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_435.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_435.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_435.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_435.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_435 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_435 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_435
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_435.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_436_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_436_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_436_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_436_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_436_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_436_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_436_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_436_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_436.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_436.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_436.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_436_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_436_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_436.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_436 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_436 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_436
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_436.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_439.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_439.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_439.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_439.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_439 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_439 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_439
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_439.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_440_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_440_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_440_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_440_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_440_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_440_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_440_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_440_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_440.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_440.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_440.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_440_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_440_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_440.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_440 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_440 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_440
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_440.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_443.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_443.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_443.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_443.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_443 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_443 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_443
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_443.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_444_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_444_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_444_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_444_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_444_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_444_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_444_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_444_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_444.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_444.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_444.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_444_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_444_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_444.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_444 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_444 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_444
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_444.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_447.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_447.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_447.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_447.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_447 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_447 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_447
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_447.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_448_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_448_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_448_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_448_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_448_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_448_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_448_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_448_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_448.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_448.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_448.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_448_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_448_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_448.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_448 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_448 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_448
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_448.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_451.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_451.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_451.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_451.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_451 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_451 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_451
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_451.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_452_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_452_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_452_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_452_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_452_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_452_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_452_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_452_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_452.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_452.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_452.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_452_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_452_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_452.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_452 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_452 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_452
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_452.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_455.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_455.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_455.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_455.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_455 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_455 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_455
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_455.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_456_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_456_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_456_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_456_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_456_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_456_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_456_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_456_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_456.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_456.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_456.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_456_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_456_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_456.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_456 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_456 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_456
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_456.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_459.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_459.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_459.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_459.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_459 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_459 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_459
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_459.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_460_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_460_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_460_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_460_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_460_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_460_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_460_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_460_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_460.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_460.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_460.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_460_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_460_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_460.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_460 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_460 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_460
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_460.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_463.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_463.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_463.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_463.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_463 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_463 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_463
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_463.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_464_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_464_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_464_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_464_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_464_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_464_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_464_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_464_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_464.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_464.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_464.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_464_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_464_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_464.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_464 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_464 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_464
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_464.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_467.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_467.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_467.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_467.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_467 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_467 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_467
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_467.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_468_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_468_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_468_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_468_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_468_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_468_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_468_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_468_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_468.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_468.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_468.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_468_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_468_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_468.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_468 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_468 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_468
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_468.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_471.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_471.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_471.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_471.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_471 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_471 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_471
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_471.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_472_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_472_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_472_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_472_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_472_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_472_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_472_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_472_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_472.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_472.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_472.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_472_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_472_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_472.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_472 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_472 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_472
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_472.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_475.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_475.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_475.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_475.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_475 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_475 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_475
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_475.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_476_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_476_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_476_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_476_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_476.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_476.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_476.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_476_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_476.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_476 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_476 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_476
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_476.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_479.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_479.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_479.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_479.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_479 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_479 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_479
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_479.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_480_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_480_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_480_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_480_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_480_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_480_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_480_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_480_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_480.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_480.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_480.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_480_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_480_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_480.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_480 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_480 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_480
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_480.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_483.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_483.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_483.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_483.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_483 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_483 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_483
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_483.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_484_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_484_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_484_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_484_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_484_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_484_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_484_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_484_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_484.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_484.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_484.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_484_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_484_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_484.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_484 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_484 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_484
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_484.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_487.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_487.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_487.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_487.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_487 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_487 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_487
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_487.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_488_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_488_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_488_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_488_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_488_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_488_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_488_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_488_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_488.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_488.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_488.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_488_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_488_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_488.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_488 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_488 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_488
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_488.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_491.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_491.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_491.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_491.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_491 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_491 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_491
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_491.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_492_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_492_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_492_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_492_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_492_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_492_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_492_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_492_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_492.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_492.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_492.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_492_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_492_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_492.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_492 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_492 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_492
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_492.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_495.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_495.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_495.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_495.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_495 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_495 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_495
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_495.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_496_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_496_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_496_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_496_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_496_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_496_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_496_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_496_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_496.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_496.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_496.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_496_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_496_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_496.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_496 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_496 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_496
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_496.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_499.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_499.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_499.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_499.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_499 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_499 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_499
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_499.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_500_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_500_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_500_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_500_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_500_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_500_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_500_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_500_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_500.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_500.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_500.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_500_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_500_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_500.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_500 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_500 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_500
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_500.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_503.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_503.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_503.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_503.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_503 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_503 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_503
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_503.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_504_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_504_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_504_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_504_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_504_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_504_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_504_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_504_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_504.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_504.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_504.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_504_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_504_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_504.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_504 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_504 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_504
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_504.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_507.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_507.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_507.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_507.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_507 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_507 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_507
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_507.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_508_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_508_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_508_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_508_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_508_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_508_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_508_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_508_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_508.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_508.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_508.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_508_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_508_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_508.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_508 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_508 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_508
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_508.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_511.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_511.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_511.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_511.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_511 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_511 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_511
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_511.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_512_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_512_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_512_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_512_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_512_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_512_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_512_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_512_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_512.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_512.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_512.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_512_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_512_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_512.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_512 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_512 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_512
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_512.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_515.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_515.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_515.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_515.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_515 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_515 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_515
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_515.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_516_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_516_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_516_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_516_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_516_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_516_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_516_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_516_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_516.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_516.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_516.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_516_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_516_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_516.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_516 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_516 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_516
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_516.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_519.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_519.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_519.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_519.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_519 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_519 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_519
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_519.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_520_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_520_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_520_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_520_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_520.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_520.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_520.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_520_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_520.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_520 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_520 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_520
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_520.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_523.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_523.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_523.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_523.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_523 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_523 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_523
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_523.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_524_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_524_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_524_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_524_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_524_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_524_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_524_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_524_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_524.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_524.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_524.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_524_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_524_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_524.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_524 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_524 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_524
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_524.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_527.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_527.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_527.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_527.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_527 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_527 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_527
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_527.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_528_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_528_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_528_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_528_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_528_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_528_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_528_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_528_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_528.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_528.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_528.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_528_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_528_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_528.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_528 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_528 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_528
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_528.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_531.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_531.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_531.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_531.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_531 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_531 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_531
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_531.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_532_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_532_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_532_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_532_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_532_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_532_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_532_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_532_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_532.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_532.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_532.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_532_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_532_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_532.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_532 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_532 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_532
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_532.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_535.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_535.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_535.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_535.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_535 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_535 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_535
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_535.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_536_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_536_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_536_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_536_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_536_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_536_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_536_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_536_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_536.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_536.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_536.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_536_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_536_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_536.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_536 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_536 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_536
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_536.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_539.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_539.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_539.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_539.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_539 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_539 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_539
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_539.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_540_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_540_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_540_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_540_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_540_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_540_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_540_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_540_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_540.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_540.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_540.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_540_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_540_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_540.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_540 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_540 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_540
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_540.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_543.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_543.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_543.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_543.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_543 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_543 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_543
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_543.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_544_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_544_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_544_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_544_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_544_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_544_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_544_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_544_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_544.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_544.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_544.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_544_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_544_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_544.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_544 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_544 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_544
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_544.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_547.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_547.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_547.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_547.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_547 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_547 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_547
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_547.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_548_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_548_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_548_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_548_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_548_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_548_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_548_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_548_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_548.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_548.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_548.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_548_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_548_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_548.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_548 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_548 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_548
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_548.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_551.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_551.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_551.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_551.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_551 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_551 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_551
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_551.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_552_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_552_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_552_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_552_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_552_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_552_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_552_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_552_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_552.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_552.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_552.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_552_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_552_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_552.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_552 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_552 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_552
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_552.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_555.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_555.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_555.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_555.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_555 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_555 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_555
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_555.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_556_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_556_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_556_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_556_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_556_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_556_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_556_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_556_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_556.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_556.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_556.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_556_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_556_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_556.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_556 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_556 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_556
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_556.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_559.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_559.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_559.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_559.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_559 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_559 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_559
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_559.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_560_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_560_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_560_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_560_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_560_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_560_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_560_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_560_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_560.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_560.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_560.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_560_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_560_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_560.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_560 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_560 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_560
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_560.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_563.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_563.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_0_0.parallel_generatedRoot_563.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_563.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_563 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_563 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_563
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_0_0.parallel_generatedRoot_563.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_564_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_564_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_564_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_564_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_564.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_564.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_564.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_564_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0_Part_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_564.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_564 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_564 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_564
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_564.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_565_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_565_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_565_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_565_10
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_565.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_565.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_565.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_0_0_Part_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_565_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_565.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_565 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_565 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_565
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_565.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_566_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_566_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_566_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_566_10
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_566.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_566.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_566.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_0_0_Part_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_566_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_566.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_566 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_566 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_566
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_566.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_567.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_567 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_567 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_567
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_580.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_580 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_580 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_580
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_582.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_582 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_582 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_582
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_584.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_584 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_584 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_584
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_586.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_586 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_586 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_586
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_588.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_588 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_588 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_588
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_590.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_590 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_590 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_590
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_592.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_592 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_592 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_592
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_594.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_594 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_594 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_594
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_596.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_596 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_596 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_596
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_598.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_598 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_598 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_598
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_600.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_600 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_600 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_600
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_602.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_602 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_602 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_602
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_604.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_604 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_604 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_604
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_606.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_606 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_606 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_606
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_608.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_608 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_608 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_608
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_610.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_610 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_610 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_610
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_612.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_612 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_612 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_612
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_614.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_614 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_614 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_614
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_616.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_616 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_616 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_616
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_618.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_618 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_618 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_618
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_620.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_620 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_620 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_620
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_622.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_622 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_622 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_622
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_624.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_624 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_624 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_624
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_626.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_626 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_626 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_626
  exact h
