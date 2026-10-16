import Stage_initial
import W_Stage_initial
import ExponentSampling

/-! Every run of the average-case relations of `Stage_initial` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_initial.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_initial.parallel_generatedRoot_10.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_10.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_10.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_10.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_10 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_10 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_10
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_10.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_11.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_11.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_11.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_11.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_11 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_11 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_11
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_11.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_12.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_12.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_12.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_12.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_12 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_12 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_12
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_12.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_13.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_13.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_13.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_13.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_13 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_13 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_13
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_13.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_14.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_14.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_14.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_14.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_14 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_14 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_14
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_14.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_15.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_15.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_15.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_15.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_15 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_15 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_15
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_15.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_16.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_16.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_16.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_16.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_16 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_16 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_16
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_16.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_17.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_17.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_17.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_17.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_17 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_17 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_17
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_17.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_18.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_18.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_18.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_18.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_18 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_18 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_18
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_18.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_19.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_19.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_19.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_19.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_19 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_19 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_19
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_19.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_20.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_20.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_20.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_20.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_20 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_20 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_20
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_20.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_21.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_21.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_21.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_21.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_21 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_21 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_21
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_21.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_22.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_22.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_22.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_22.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_22 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_22 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_22
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_22.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_23.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_23.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_23.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_23.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_23 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_23 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_23
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_23.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_24.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_24.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_24.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_24.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_24 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_24 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_24
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_24.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_25.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_25.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_25.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_25.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_25 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_25 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_25
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_25.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_26.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_26.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_26.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_26.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_26 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_26 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_26
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_26.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_27.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_27.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_27.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_27.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_27 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_27 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_27
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_27.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_28.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_28.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_28.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_28.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_28 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_28 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_28
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_28.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_29.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_29.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_29.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_29.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_29 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_29 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_29
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_29.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_30.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_30.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_30.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_30.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_30 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_30 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_30
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_30.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_31.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_31.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_31.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_31.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_31 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_31 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_31
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_31.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_32.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_32.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_32.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_32.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_32 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_32 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_32
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_32.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_33.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_33.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_33.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_33.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_33 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_33 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_33
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_33.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_34.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_34.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_34.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_34.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_34 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_34 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_34
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_34.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_35.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_35.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_35.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_35.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_35 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_35 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_35
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_35.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_36.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_36.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_36.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_36.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_36 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_36 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_36
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_36.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_37.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_37.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_37.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_37.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_37 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_37 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_37
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_37.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_38.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_38.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_38.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_38.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_38 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_38 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_38
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_38.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_39.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_39.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_39.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_39.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_39 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_39 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_39
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_39.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_40.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_40.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_40.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_40.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_40 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_40 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_40
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_40.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_41.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_41.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_41.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_41.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_41 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_41 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_41
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_41.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_42.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_42.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_42.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_42.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_42 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_42 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_42
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_42.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_43.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_43.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_43.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_43.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_43 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_43 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_43
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_43.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_44.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_44.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_44.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_44.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_44 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_44 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_44
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_44.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_45.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_45.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_45.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_45.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_45 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_45 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_45
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_45.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_46.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_46.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_46.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_46.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_46 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_46 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_46
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_46.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_47.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_47.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_47.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_47.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_47 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_47 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_47
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_47.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_48.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_48.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_48.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_48.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_48 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_48 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_48
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_48.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_49.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_49.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_49.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_49.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_49 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_49 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_49
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_49.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_50.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_50.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_50.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_50.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_50 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_50 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_50
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_50.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_51.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_51.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_51.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_51.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_51 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_51 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_51
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_51.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_52.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_52.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_52.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_52.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_52 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_52 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_52
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_52.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_53.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_53.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_53.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_53.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_53 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_53 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_53
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_53.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_54.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_54.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_54.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_54.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_54 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_54 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_54
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_54.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_55.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_55.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_55.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_55.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_55 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_55 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_55
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_55.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_56.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_56.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_56.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_56.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_56 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_56 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_56
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_56.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_57.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_57.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_57.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_57.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_57 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_57 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_57
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_57.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_58.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_58.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_58.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_58.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_58 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_58 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_58
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_58.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_59.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_59.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_59.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_59.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_59 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_59 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_59
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_59.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_60.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_60.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_60.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_60.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_60 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_60 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_60
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_60.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_61.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_61.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_61.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_61.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_61 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_61 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_61
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_61.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_62.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_62.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_62.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_62.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_62 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_62 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_62
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_62.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_63.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_63.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_63.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_63.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_63 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_63 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_63
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_63.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_64.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_64.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_64.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_64.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_64 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_64 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_64
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_64.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_65.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_65.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_65.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_65.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_65 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_65 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_65
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_65.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_66.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_66.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_66.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_66.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_66 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_66 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_66
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_66.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_67.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_67.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_67.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_67.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_67 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_67 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_67
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_67.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_68.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_68.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_68.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_68.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_68 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_68 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_68
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_68.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_69.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_69.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_69.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_69.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_69 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_69 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_69
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_69.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_70.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_70.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_70.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_70.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_70 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_70 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_70
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_70.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_71.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_71.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_71.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_71.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_71 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_71 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_71
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_71.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_72.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_72.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_72.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_72.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_72 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_72 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_72
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_72.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_73.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_73.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_73.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_73.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_73 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_73 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_73
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_73.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_74.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_74.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_74.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_74.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_74 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_74 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_74
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_74.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_75.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_75.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_75.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_75.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_75 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_75 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_75
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_75.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_76.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_76.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_76.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_76.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_76 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_76 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_76
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_76.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_77.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_77.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_77.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_77.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_77 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_77 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_77
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_77.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_78.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_78.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_78.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_78.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_78 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_78 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_78
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_78.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_79.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_79.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_79.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_79.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_79 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_79 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_79
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_79.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_80.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_80.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_80.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_80.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_80 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_80 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_80
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_80.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_81.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_81.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_81.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_81.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_81 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_81 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_81
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_81.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_82.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_82.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_82.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_82.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_82 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_82 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_82
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_82.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_83.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_83.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_83.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_83.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_83 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_83 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_83
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_83.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_84.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_84.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_84.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_84.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_84 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_84 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_84
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_84.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_85.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_85.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_85.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_85.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_85 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_85 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_85
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_85.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_86.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_86.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_86.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_86.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_86 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_86 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_86
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_86.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_87.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_87.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_87.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_87.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_87 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_87 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_87
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_87.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_88.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_88.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_88.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_88.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_88 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_88 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_88
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_88.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_89.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_89.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_89.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_89.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_89 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_89 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_89
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_89.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_90.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_90.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_90.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_90.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_90 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_90 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_90
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_90.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_91.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_91.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_91.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_91.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_91 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_91 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_91
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_91.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_92.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_92.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_92.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_92.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_92 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_92 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_92
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_92.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_93.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_93.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_93.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_93.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_93 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_93 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_93
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_93.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_94.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_94.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_94.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_94.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_94 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_94 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_94
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_94.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_95.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_95.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_95.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_95.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_95 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_95 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_95
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_95.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_96.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_96.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_96.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_96.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_96 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_96 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_96
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_96.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_97.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_97.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_97.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_97.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_97 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_97 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_97
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_97.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_98.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_98.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_98.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_98.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_98 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_98 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_98
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_98.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_99.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_99.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_99.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_99.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_99 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_99 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_99
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_99.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_100.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_100.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_100.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_100.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_100 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_100 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_100
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_100.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_101.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_101.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_101.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_101.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_101 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_101 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_101
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_101.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_102.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_102.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_102.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_102.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_102 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_102 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_102
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_102.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_103.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_103.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_103.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_103.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_103 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_103 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_103
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_103.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_104.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_104.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_104.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_104.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_104 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_104 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_104
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_104.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_105.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_105.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_105.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_105.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_105 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_105 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_105
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_105.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_106.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_106.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_106.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_106.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_106 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_106 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_106
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_106.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_107.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_107.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_107.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_107.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_107 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_107 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_107
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_107.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_108.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_108.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_108.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_108.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_108 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_108 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_108
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_108.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_109.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_109.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_109.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_109.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_109 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_109 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_109
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_109.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_110.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_110.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_110.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_110.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_110 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_110 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_110
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_110.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_111.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_111.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_111.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_111.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_111 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_111 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_111
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_111.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_112.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_112.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_112.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_112.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_112 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_112 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_112
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_112.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_113.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_113.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_113.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_113.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_113 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_113 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_113
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_113.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_114.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_114.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_114.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_114.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_114 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_114 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_114
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_114.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_115.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_115.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_115.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_115.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_115 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_115 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_115
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_115.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_116.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_116.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_116.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_116.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_116 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_116 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_116
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_116.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_117.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_117.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_117.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_117.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_117 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_117 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_117
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_117.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_118.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_118.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_118.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_118.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_118 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_118 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_118
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_118.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_119.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_119.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_119.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_119.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_119 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_119 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_119
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_119.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_120.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_120.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_120.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_120.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_120 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_120 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_120
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_120.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_121.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_121.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_121.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_121.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_121 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_121 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_121
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_121.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_122.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_122.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_122.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_122.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_122 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_122 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_122
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_122.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_123.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_123.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_123.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_123.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_123 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_123 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_123
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_123.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_124.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_124.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_124.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_124.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_124 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_124 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_124
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_124.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_125.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_125.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_125.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_125.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_125 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_125 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_125
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_125.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_126.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_126.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_126.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_126.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_126 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_126 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_126
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_126.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_127.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_127.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_127.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_127.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_127 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_127 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_127
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_127.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_128.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_128.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_128.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_128.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_128 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_128 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_128
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_128.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_129.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_129.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_129.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_129.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_129 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_129 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_129
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_129.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_130.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_130.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_130.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_130.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_130 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_130 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_130
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_130.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_131.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_131.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_131.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_131.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_131 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_131 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_131
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_131.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_132.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_132.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_132.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_132.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_132 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_132 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_132
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_132.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_133.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_133.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_133.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_133.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_133 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_133 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_133
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_133.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_134.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_134.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_134.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_134.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_134 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_134 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_134
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_134.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_135.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_135.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_135.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_135.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_135 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_135 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_135
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_135.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_136.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_136.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_136.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_136.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_136 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_136 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_136
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_136.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_137.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_137.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_137.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_137.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_137 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_137 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_137
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_137.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_138.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_138.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_138.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_138.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_138 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_138 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_138
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_138.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_139.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_139.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_139.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_139.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_139 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_139 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_139
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_139.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_140.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_140.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_140.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_140.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_140 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_140 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_140
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_140.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_141.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_141.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_141.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_141.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_141 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_141 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_141
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_141.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_142.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_142.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_142.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_142.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_142 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_142 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_142
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_142.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_143.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_143.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_143.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_143.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_143 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_143 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_143
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_143.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_144.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_144.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_144.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_144.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_144 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_144 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_144
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_144.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_145.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_145.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_145.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_145.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_145 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_145 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_145
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_145.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_146.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_146.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_146.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_146.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_146 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_146 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_146
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_146.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_147.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_147.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_147.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_147.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_147 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_147 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_147
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_147.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_148.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_148.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_148.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_148.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_148 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_148 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_148
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_148.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_149.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_149.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_149.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_149.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_149 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_149 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_149
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_149.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_150.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_150.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_150.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_150.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_150 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_150 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_150
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_150.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_151.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_151.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_151.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_151.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_151 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_151 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_151
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_151.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_152.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_152.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_152.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_152.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_152 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_152 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_152
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_152.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_153.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_153.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_153.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_153.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_153 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_153 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_153
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_153.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_154.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_154.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_154.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_154.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_154 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_154 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_154
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_154.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_155.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_155.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_155.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_155.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_155 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_155 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_155
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_155.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_156.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_156.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_156.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_156.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_156 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_156 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_156
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_156.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_157.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_157.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_157.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_157.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_157 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_157 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_157
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_157.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_158.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_158.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_158.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_158.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_158 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_158 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_158
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_158.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_159.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_159.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_159.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_159.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_159 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_159 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_159
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_159.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_160.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_160.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_160.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_160.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_160 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_160 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_160
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_160.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_161.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_161.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_161.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_161.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_161 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_161 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_161
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_161.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_162.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_162.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_162.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_162.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_162 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_162 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_162
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_162.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_163.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_163.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_163.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_163.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_163 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_163 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_163
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_163.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_164.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_164.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_164.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_164.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_164 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_164 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_164
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_164.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_165.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_165.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_165.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_165.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_165 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_165 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_165
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_165.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_166.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_166.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_166.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_166.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_166 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_166 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_166
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_166.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_167.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_167.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_167.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_167.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_167 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_167 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_167
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_167.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_168.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_168.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_168.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_168.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_168 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_168 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_168
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_168.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_169.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_169.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_169.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_169.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_169 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_169 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_169
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_169.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_170.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_170.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_170.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_170.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_170 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_170 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_170
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_170.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_171.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_171.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_171.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_171.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_171 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_171 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_171
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_171.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_172.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_172.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_172.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_172.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_172 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_172 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_172
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_172.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_173.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_173.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_173.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_173.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_173 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_173 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_173
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_173.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_174.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_174.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_174.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_174.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_174 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_174 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_174
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_174.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_175.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_175.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_175.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_175.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_175 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_175 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_175
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_175.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_176.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_176.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_176.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_176.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_176 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_176 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_176
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_176.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_177.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_177.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_177.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_177.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_177 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_177 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_177
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_177.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_178.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_178.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_178.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_178.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_178 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_178 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_178
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_178.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_179.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_179.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_179.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_179.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_179 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_179 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_179
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_179.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_180.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_180.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_180.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_180.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_180 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_180 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_180
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_180.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_181.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_181.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_181.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_181.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_181 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_181 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_181
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_181.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_182.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_182.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_182.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_182.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_182 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_182 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_182
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_182.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_183.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_183.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_183.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_183.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_183 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_183 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_183
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_183.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_184.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_184.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_184.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_184.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_184 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_184 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_184
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_184.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_185.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_185.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_185.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_185.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_185 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_185 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_185
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_185.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_186.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_186.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_186.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_186.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_186 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_186 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_186
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_186.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_187.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_187.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_187.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_187.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_187 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_187 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_187
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_187.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_188.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_188.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_188.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_188.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_188 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_188 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_188
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_188.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_189.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_189.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_189.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_189.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_189 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_189 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_189
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_189.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_190.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_190.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_190.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_190.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_190 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_190 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_190
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_190.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_191.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_191.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_191.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_191.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_191 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_191 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_191
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_191.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_192.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_192.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_192.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_192.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_192 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_192 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_192
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_192.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_193.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_193.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_193.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_193.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_193 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_193 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_193
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_193.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_194.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_194.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_194.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_194.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_194 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_194 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_194
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_194.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_195.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_195.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_195.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_195.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_195 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_195 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_195
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_195.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_196.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_196.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_196.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_196.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_196 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_196 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_196
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_196.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_197.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_197.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_197.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_197.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_197 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_197 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_197
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_197.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_198.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_198.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_198.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_198.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_198 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_198 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_198
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_198.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_199.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_199.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_199.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_199.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_199 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_199 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_199
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_199.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_200.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_200.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_200.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_200.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_200 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_200 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_200
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_200.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_201.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_201.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_201.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_201.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_201 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_201 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_201
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_201.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_202.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_202.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_202.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_202.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_202 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_202 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_202
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_202.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_203.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_203.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_203.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_203.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_203 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_203 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_203
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_203.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_204.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_204.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_204.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_204.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_204 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_204 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_204
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_204.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_205.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_205.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_205.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_205.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_205 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_205 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_205
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_205.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_206.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_206.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_206.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_206.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_206 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_206 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_206
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_206.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_207.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_207.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_207.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_207.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_207 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_207 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_207
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_207.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_208.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_208.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_208.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_208.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_208 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_208 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_208
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_208.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_209.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_209.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_209.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_209.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_209 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_209 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_209
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_209.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_210.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_210.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_210.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_210.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_210 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_210 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_210
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_210.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_211.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_211.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_211.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_211.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_211 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_211 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_211
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_211.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_212.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_212.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_212.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_212.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_212 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_212 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_212
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_212.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_213.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_213.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_213.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_213.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_213 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_213 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_213
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_213.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_214.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_214.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_214.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_214.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_214 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_214 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_214
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_214.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_215.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_215.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_215.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_215.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_215 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_215 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_215
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_215.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_216.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_216.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_216.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_216.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_216 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_216 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_216
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_216.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_217.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_217.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_217.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_217.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_217 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_217 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_217
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_217.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_218.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_218.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_218.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_218.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_218 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_218 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_218
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_218.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_219.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_219.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_219.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_219.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_219 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_219 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_219
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_219.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_220.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_220.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_220.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_220.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_220 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_220 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_220
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_220.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_221.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_221.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_221.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_221.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_221 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_221 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_221
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_221.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_222.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_222.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_222.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_222.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_222 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_222 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_222
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_222.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_223.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_223.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_223.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_223.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_223 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_223 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_223
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_223.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_224.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_224.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_224.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_224.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_224 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_224 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_224
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_224.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_225.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_225.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_225.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_225.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_225 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_225 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_225
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_225.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_226.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_226.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_226.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_226.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_226 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_226 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_226
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_226.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_227.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_227.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_227.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_227.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_227 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_227 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_227
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_227.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_228.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_228.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_228.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_228.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_228 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_228 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_228
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_228.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_229.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_229.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_229.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_229.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_229 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_229 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_229
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_229.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_230.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_230.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_230.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_230.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_230 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_230 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_230
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_230.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_231.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_231.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_231.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_231.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_231 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_231 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_231
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_231.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_232.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_232.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_232.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_232.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_232 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_232 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_232
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_232.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_233.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_233.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_233.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_233.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_233 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_233 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_233
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_233.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_234.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_234.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_234.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_234.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_234 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_234 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_234
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_234.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_235.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_235.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_235.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_235.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_235 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_235 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_235
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_235.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_236.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_236.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_236.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_236.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_236 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_236 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_236
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_236.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_237.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_237.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_237.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_237.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_237 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_237 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_237
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_237.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_238.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_238.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_238.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_238.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_238 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_238 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_238
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_238.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_239.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_239.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_239.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_239.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_239 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_239 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_239
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_239.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_240.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_240.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_240.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_240.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_240 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_240 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_240
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_240.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_241.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_241.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_241.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_241.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_241 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_241 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_241
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_241.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_242.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_242.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_242.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_242.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_242 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_242 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_242
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_242.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_243.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_243.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_243.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_243.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_243 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_243 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_243
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_243.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_244.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_244.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_244.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_244.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_244 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_244 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_244
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_244.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_245.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_245.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_245.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_245.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_245 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_245 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_245
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_245.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_246.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_246.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_246.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_246.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_246 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_246 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_246
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_246.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_247.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_247.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_247.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_247.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_247 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_247 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_247
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_247.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_248.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_248.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_248.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_248.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_248 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_248 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_248
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_248.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_249.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_249.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_249.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_249.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_249 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_249 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_249
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_249.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_250.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_250.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_250.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_250.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_250 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_250 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_250
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_250.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_251.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_251.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_251.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_251.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_251 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_251 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_251
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_251.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_252.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_252.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_252.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_252.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_252 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_252 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_252
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_252.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_253.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_253.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_253.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_253.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_253 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_253 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_253
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_253.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_254.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_254.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_254.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_254.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_254 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_254 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_254
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_254.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_255.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_255.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_255.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_255.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_255 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_255 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_255
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_255.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_256.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_256.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_256.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_256.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_256 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_256 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_256
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_256.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_257.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_257.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_257.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_257.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_257 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_257 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_257
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_257.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_258.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_258.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_258.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_258.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_258 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_258 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_258
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_258.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_259.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_259.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_259.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_259.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_259 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_259 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_259
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_259.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_260.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_260.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_260.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_260.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_260 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_260 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_260
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_260.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_261.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_261.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_261.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_261.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_261 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_261 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_261
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_261.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_262.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_262.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_262.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_262.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_262 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_262 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_262
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_262.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_263.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_263.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_263.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_263.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_263 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_263 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_263
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_263.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_264.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_264.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_264.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_264.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_264 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_264 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_264
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_264.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_265.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_265.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_265.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_265.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_265 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_265 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_265
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_265.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_266.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_266.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_266.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_266.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_266 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_266 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_266
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_266.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_267.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_267.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_267.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_267.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_267 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_267 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_267
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_267.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_268.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_268.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_268.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_268.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_268 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_268 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_268
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_268.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_269.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_269.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_269.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_269.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_269 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_269 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_269
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_269.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_270.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_270.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_270.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_270.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_270 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_270 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_270
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_270.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_271.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_271.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_271.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_271.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_271 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_271 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_271
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_271.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_272.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_272.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_272.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_272.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_272 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_272 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_272
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_272.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_273.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_273.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_273.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_273.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_273 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_273 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_273
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_273.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_274.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_274.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_274.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_274.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_274 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_274 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_274
  exact (by obtain ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_9_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_274.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_275.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_275.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs) : W_Stage_initial.parallel_generatedRoot_275.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h6), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_initial.parallel_generatedRoot_275.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_275 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_275 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_275
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_13_0, w_14_0, (Stage_initial.parallel_generatedRoot_275.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_277.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_277.constraints_0 backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs) : W_Stage_initial.parallel_generatedRoot_277.constraints_0 backend hashModel i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h12), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h13), h14⟩)


theorem Stage_initial.parallel_generatedRoot_277.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_277 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_277 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_277
  exact (by obtain ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, (Stage_initial.parallel_generatedRoot_277.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_278.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_278.constraints_0 backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs) : W_Stage_initial.parallel_generatedRoot_278.constraints_0 backend hashModel i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h12), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h13), h14⟩)


theorem Stage_initial.parallel_generatedRoot_278.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_278 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_278 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_278
  exact (by obtain ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, (Stage_initial.parallel_generatedRoot_278.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_279.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_279.constraints_0 backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs) : W_Stage_initial.parallel_generatedRoot_279.constraints_0 backend hashModel i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_279.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_279 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_279 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_279
  exact (by obtain ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, (Stage_initial.parallel_generatedRoot_279.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_280.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_280.constraints_0 backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs) : W_Stage_initial.parallel_generatedRoot_280.constraints_0 backend hashModel i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_280.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_280 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_280 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_280
  exact (by obtain ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, (Stage_initial.parallel_generatedRoot_280.constraints_0.bridge h1)⟩)


theorem Stage_initial.generatedRoot.constraints_17.bridge {backend hashModel tape path w_9_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_283_0 w_289_0 w_292_0 w_293_0 w_295_0 w_296_0 w_298_0 w_304_0 w_307_0 w_308_0 w_310_0 w_311_0 w_313_0 w_322_0 w_325_0 w_326_0 w_328_0 w_329_0 w_331_0 w_339_0 w_342_0 w_343_0 w_345_0 outputs}
    (h : Stage_initial.generatedRoot.constraints_17 backend hashModel tape path w_9_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_283_0 w_289_0 w_292_0 w_293_0 w_295_0 w_296_0 w_298_0 w_304_0 w_307_0 w_308_0 w_310_0 w_311_0 w_313_0 w_322_0 w_325_0 w_326_0 w_328_0 w_329_0 w_331_0 w_339_0 w_342_0 w_343_0 w_345_0 outputs) : W_Stage_initial.generatedRoot.constraints_17 backend hashModel w_9_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_283_0 w_289_0 w_292_0 w_293_0 w_295_0 w_296_0 w_298_0 w_304_0 w_307_0 w_308_0 w_310_0 w_311_0 w_313_0 w_322_0 w_325_0 w_326_0 w_328_0 w_329_0 w_331_0 w_339_0 w_342_0 w_343_0 w_345_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20⟩ := h; exact ⟨h1, h2, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h3), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h4), h5, h6, h7, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h8), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), h10, h11, h12, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h13), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h14), h15, h16, h17, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h18), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h19), h20⟩)


theorem Stage_initial.generatedRoot.constraints_16.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_276_0 w_1_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_281_0}
    (h : Stage_initial.generatedRoot.constraints_16 backend hashModel tape path params w_0_0 w_9_0 w_276_0 w_1_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_281_0) : W_Stage_initial.generatedRoot.constraints_16 backend hashModel params' w_0_0 w_9_0 w_276_0 w_1_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_281_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_264.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_265.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_266.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_267.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_268.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_269.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_270.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_271.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_272.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_273.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_274.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_275.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_277.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_278.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_279.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_280.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_15.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0}
    (h : Stage_initial.generatedRoot.constraints_15 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0) : W_Stage_initial.generatedRoot.constraints_15 backend hashModel params' w_0_0 w_9_0 w_1_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_248.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_249.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_250.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_251.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_252.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_253.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_254.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_255.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_256.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_257.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_258.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_259.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_260.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_261.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_262.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_263.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_14.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0}
    (h : Stage_initial.generatedRoot.constraints_14 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0) : W_Stage_initial.generatedRoot.constraints_14 backend hashModel params' w_0_0 w_9_0 w_1_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_232.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_233.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_234.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_235.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_236.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_237.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_238.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_239.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_240.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_241.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_242.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_243.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_244.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_245.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_246.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_247.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_13.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0}
    (h : Stage_initial.generatedRoot.constraints_13 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0) : W_Stage_initial.generatedRoot.constraints_13 backend hashModel params' w_0_0 w_9_0 w_1_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_216.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_217.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_218.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_219.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_220.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_221.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_222.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_223.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_224.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_225.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_226.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_227.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_228.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_229.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_230.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_231.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_12.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0}
    (h : Stage_initial.generatedRoot.constraints_12 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0) : W_Stage_initial.generatedRoot.constraints_12 backend hashModel params' w_0_0 w_9_0 w_1_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_200.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_201.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_202.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_203.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_204.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_205.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_206.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_207.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_208.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_209.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_210.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_211.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_212.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_213.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_214.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_215.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_11.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0}
    (h : Stage_initial.generatedRoot.constraints_11 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0) : W_Stage_initial.generatedRoot.constraints_11 backend hashModel params' w_0_0 w_9_0 w_1_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_184.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_185.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_186.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_187.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_188.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_189.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_190.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_191.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_192.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_193.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_194.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_195.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_196.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_197.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_198.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_199.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_10.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0}
    (h : Stage_initial.generatedRoot.constraints_10 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0) : W_Stage_initial.generatedRoot.constraints_10 backend hashModel params' w_0_0 w_9_0 w_1_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_168.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_169.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_170.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_171.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_172.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_173.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_174.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_175.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_176.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_177.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_178.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_179.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_180.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_181.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_182.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_183.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_9.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0}
    (h : Stage_initial.generatedRoot.constraints_9 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0) : W_Stage_initial.generatedRoot.constraints_9 backend hashModel params' w_0_0 w_9_0 w_1_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_152.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_153.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_154.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_155.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_156.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_157.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_158.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_159.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_160.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_161.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_162.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_163.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_164.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_165.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_166.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_167.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_8.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0}
    (h : Stage_initial.generatedRoot.constraints_8 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0) : W_Stage_initial.generatedRoot.constraints_8 backend hashModel params' w_0_0 w_9_0 w_1_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_136.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_137.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_138.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_139.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_140.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_141.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_142.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_143.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_144.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_145.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_146.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_147.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_148.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_149.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_150.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_151.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_7.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0}
    (h : Stage_initial.generatedRoot.constraints_7 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0) : W_Stage_initial.generatedRoot.constraints_7 backend hashModel params' w_0_0 w_9_0 w_1_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_120.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_121.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_122.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_123.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_124.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_125.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_126.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_127.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_128.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_129.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_130.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_131.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_132.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_133.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_134.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_135.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_6.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0}
    (h : Stage_initial.generatedRoot.constraints_6 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0) : W_Stage_initial.generatedRoot.constraints_6 backend hashModel params' w_0_0 w_9_0 w_1_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_104.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_105.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_106.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_107.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_108.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_109.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_110.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_111.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_112.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_113.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_114.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_115.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_116.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_117.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_118.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_119.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_5.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0}
    (h : Stage_initial.generatedRoot.constraints_5 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0) : W_Stage_initial.generatedRoot.constraints_5 backend hashModel params' w_0_0 w_9_0 w_1_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_88.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_89.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_90.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_91.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_92.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_93.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_94.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_95.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_96.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_97.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_98.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_99.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_100.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_101.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_102.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_103.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_4.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0}
    (h : Stage_initial.generatedRoot.constraints_4 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0) : W_Stage_initial.generatedRoot.constraints_4 backend hashModel params' w_0_0 w_9_0 w_1_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_72.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_73.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_74.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_75.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_76.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_77.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_78.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_79.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_80.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_81.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_82.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_83.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_84.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_85.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_86.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_87.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_3.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0}
    (h : Stage_initial.generatedRoot.constraints_3 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0) : W_Stage_initial.generatedRoot.constraints_3 backend hashModel params' w_0_0 w_9_0 w_1_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_56.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_57.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_58.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_59.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_60.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_61.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_62.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_63.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_64.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_65.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_66.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_67.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_68.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_69.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_70.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_71.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_2.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0}
    (h : Stage_initial.generatedRoot.constraints_2 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0) : W_Stage_initial.generatedRoot.constraints_2 backend hashModel params' w_0_0 w_9_0 w_1_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_40.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_41.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_42.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_43.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_44.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_45.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_46.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_47.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_48.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_49.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_50.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_51.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_52.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_53.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_54.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_55.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_1.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0}
    (h : Stage_initial.generatedRoot.constraints_1 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0) : W_Stage_initial.generatedRoot.constraints_1 backend hashModel params' w_0_0 w_9_0 w_1_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_24.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_25.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_26.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_27.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_28.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_29.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_30.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_31.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_32.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_33.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_34.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_35.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_36.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_37.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_38.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_39.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0}
    (h : Stage_initial.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0) : W_Stage_initial.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h2), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h3), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_10.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_11.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_12.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_13.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_14.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_15.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_16.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_17.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_18.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_19.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_20.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_21.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_22.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_23.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_initial.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_initial.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_initial.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_5_0, wit1.w_6_0, wit1.w_10_0, wit1.w_11_0, wit1.w_12_0, wit1.w_13_0, wit1.w_14_0, wit1.w_15_0, wit1.w_16_0, wit1.w_17_0, wit1.w_18_0, wit1.w_19_0, wit1.w_20_0, wit1.w_21_0, wit1.w_22_0, wit1.w_23_0, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_31_0, wit1.w_32_0, wit1.w_33_0, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_41_0, wit1.w_42_0, wit1.w_43_0, wit1.w_44_0, wit1.w_45_0, wit1.w_46_0, wit1.w_47_0, wit1.w_48_0, wit1.w_49_0, wit1.w_50_0, wit1.w_51_0, wit1.w_52_0, wit1.w_53_0, wit1.w_54_0, wit1.w_55_0, wit1.w_56_0, wit1.w_57_0, wit1.w_58_0, wit1.w_59_0, wit1.w_60_0, wit1.w_61_0, wit1.w_62_0, wit1.w_63_0, wit1.w_64_0, wit1.w_65_0, wit1.w_66_0, wit1.w_67_0, wit1.w_68_0, wit1.w_69_0, wit1.w_70_0, wit1.w_71_0, wit1.w_72_0, wit1.w_73_0, wit1.w_74_0, wit1.w_75_0, wit1.w_76_0, wit1.w_77_0, wit1.w_78_0, wit1.w_79_0, wit1.w_80_0, wit1.w_81_0, wit1.w_82_0, wit1.w_83_0, wit1.w_84_0, wit1.w_85_0, wit1.w_86_0, wit1.w_87_0, wit1.w_88_0, wit1.w_89_0, wit1.w_90_0, wit1.w_91_0, wit1.w_92_0, wit1.w_93_0, wit1.w_94_0, wit1.w_95_0, wit1.w_96_0, wit1.w_97_0, wit1.w_98_0, wit1.w_99_0, wit1.w_100_0, wit1.w_101_0, wit1.w_102_0, wit1.w_103_0, wit1.w_104_0, wit1.w_105_0, wit1.w_106_0, wit1.w_107_0, wit1.w_108_0, wit1.w_109_0, wit1.w_110_0, wit1.w_111_0, wit1.w_112_0, wit1.w_113_0, wit1.w_114_0, wit1.w_115_0, wit1.w_116_0, wit1.w_117_0, wit1.w_118_0, wit1.w_119_0, wit1.w_120_0, wit1.w_121_0, wit1.w_122_0, wit1.w_123_0, wit1.w_124_0, wit1.w_125_0, wit1.w_126_0, wit1.w_127_0, wit1.w_128_0, wit1.w_129_0, wit1.w_130_0, wit1.w_131_0, wit1.w_132_0, wit1.w_133_0, wit1.w_134_0, wit1.w_135_0, wit1.w_136_0, wit1.w_137_0, wit1.w_138_0, wit1.w_139_0, wit1.w_140_0, wit1.w_141_0, wit1.w_142_0, wit1.w_143_0, wit1.w_144_0, wit1.w_145_0, wit1.w_146_0, wit1.w_147_0, wit1.w_148_0, wit1.w_149_0, wit1.w_150_0, wit1.w_151_0, wit1.w_152_0, wit1.w_153_0, wit1.w_154_0, wit1.w_155_0, wit1.w_156_0, wit1.w_157_0, wit1.w_158_0, wit1.w_159_0, wit1.w_160_0, wit1.w_161_0, wit1.w_162_0, wit1.w_163_0, wit1.w_164_0, wit1.w_165_0, wit1.w_166_0, wit1.w_167_0, wit1.w_168_0, wit1.w_169_0, wit1.w_170_0, wit1.w_171_0, wit1.w_172_0, wit1.w_173_0, wit1.w_174_0, wit1.w_175_0, wit1.w_176_0, wit1.w_177_0, wit1.w_178_0, wit1.w_179_0, wit1.w_180_0, wit1.w_181_0, wit1.w_182_0, wit1.w_183_0, wit1.w_184_0, wit1.w_185_0, wit1.w_186_0, wit1.w_187_0, wit1.w_188_0, wit1.w_189_0, wit1.w_190_0, wit1.w_191_0, wit1.w_192_0, wit1.w_193_0, wit1.w_194_0, wit1.w_195_0, wit1.w_196_0, wit1.w_197_0, wit1.w_198_0, wit1.w_199_0, wit1.w_200_0, wit1.w_201_0, wit1.w_202_0, wit1.w_203_0, wit1.w_204_0, wit1.w_205_0, wit1.w_206_0, wit1.w_207_0, wit1.w_208_0, wit1.w_209_0, wit1.w_210_0, wit1.w_211_0, wit1.w_212_0, wit1.w_213_0, wit1.w_214_0, wit1.w_215_0, wit1.w_216_0, wit1.w_217_0, wit1.w_218_0, wit1.w_219_0, wit1.w_220_0, wit1.w_221_0, wit1.w_222_0, wit1.w_223_0, wit1.w_224_0, wit1.w_225_0, wit1.w_226_0, wit1.w_227_0, wit1.w_228_0, wit1.w_229_0, wit1.w_230_0, wit1.w_231_0, wit1.w_232_0, wit1.w_233_0, wit1.w_234_0, wit1.w_235_0, wit1.w_236_0, wit1.w_237_0, wit1.w_238_0, wit1.w_239_0, wit1.w_240_0, wit1.w_241_0, wit1.w_242_0, wit1.w_243_0, wit1.w_244_0, wit1.w_245_0, wit1.w_246_0, wit1.w_247_0, wit1.w_248_0, wit1.w_249_0, wit1.w_250_0, wit1.w_251_0, wit1.w_252_0, wit1.w_253_0, wit1.w_254_0, wit1.w_255_0, wit1.w_256_0, wit1.w_257_0, wit1.w_258_0, wit1.w_259_0, wit1.w_260_0, wit1.w_261_0, wit1.w_262_0, wit1.w_263_0, wit1.w_264_0, wit1.w_265_0, wit1.w_266_0, wit1.w_267_0, wit1.w_268_0, wit1.w_269_0, wit1.w_270_0, wit1.w_271_0, wit1.w_272_0, wit1.w_273_0, wit1.w_274_0, wit1.w_275_0, wit1.w_277_0, wit1.w_278_0, wit1.w_279_0, wit1.w_280_0, wit1.w_281_0, wit1.w_283_0, wit1.w_289_0, wit1.w_292_0, wit1.w_293_0, wit1.w_296_0, wit1.w_298_0, wit1.w_304_0, wit1.w_307_0, wit1.w_308_0, wit1.w_311_0, wit1.w_313_0, wit1.w_322_0, wit1.w_325_0, wit1.w_326_0, wit1.w_329_0, wit1.w_331_0, wit1.w_339_0, wit1.w_342_0, wit1.w_343_0⟩, (by obtain ⟨h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20⟩ := h2; exact ⟨(Stage_initial.generatedRoot.constraints_0.bridge h3), (Stage_initial.generatedRoot.constraints_1.bridge h4), (Stage_initial.generatedRoot.constraints_2.bridge h5), (Stage_initial.generatedRoot.constraints_3.bridge h6), (Stage_initial.generatedRoot.constraints_4.bridge h7), (Stage_initial.generatedRoot.constraints_5.bridge h8), (Stage_initial.generatedRoot.constraints_6.bridge h9), (Stage_initial.generatedRoot.constraints_7.bridge h10), (Stage_initial.generatedRoot.constraints_8.bridge h11), (Stage_initial.generatedRoot.constraints_9.bridge h12), (Stage_initial.generatedRoot.constraints_10.bridge h13), (Stage_initial.generatedRoot.constraints_11.bridge h14), (Stage_initial.generatedRoot.constraints_12.bridge h15), (Stage_initial.generatedRoot.constraints_13.bridge h16), (Stage_initial.generatedRoot.constraints_14.bridge h17), (Stage_initial.generatedRoot.constraints_15.bridge h18), (Stage_initial.generatedRoot.constraints_16.bridge h19), (Stage_initial.generatedRoot.constraints_17.bridge h20)⟩)⟩)
