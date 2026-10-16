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


theorem Stage_initial.parallel_generatedRoot_29.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_29.constraints_0 backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs) : W_Stage_initial.parallel_generatedRoot_29.constraints_0 backend hashModel i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h12), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h13), h14⟩)


theorem Stage_initial.parallel_generatedRoot_29.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_29 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_29 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_29
  exact (by obtain ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, (Stage_initial.parallel_generatedRoot_29.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_30.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_30.constraints_0 backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs) : W_Stage_initial.parallel_generatedRoot_30.constraints_0 backend hashModel i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h12), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h13), h14⟩)


theorem Stage_initial.parallel_generatedRoot_30.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_30 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_30 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_30
  exact (by obtain ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_18_0, w_24_0, w_27_0, w_30_0, w_31_0, (Stage_initial.parallel_generatedRoot_30.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_31.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_31.constraints_0 backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs) : W_Stage_initial.parallel_generatedRoot_31.constraints_0 backend hashModel i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_31.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_31 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_31 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_31
  exact (by obtain ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, (Stage_initial.parallel_generatedRoot_31.constraints_0.bridge h1)⟩)


theorem Stage_initial.parallel_generatedRoot_32.constraints_0.bridge {backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs}
    (h : Stage_initial.parallel_generatedRoot_32.constraints_0 backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs) : W_Stage_initial.parallel_generatedRoot_32.constraints_0 backend hashModel i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h9), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h10), h11⟩)


theorem Stage_initial.parallel_generatedRoot_32.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_initial.parallel_generatedRoot_32 backend hashModel tape path params i_0 inputs outputs) : W_Stage_initial.parallel_generatedRoot_32 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_initial.parallel_generatedRoot_32
  exact (by obtain ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_20_0, w_25_0, w_28_0, w_29_0, (Stage_initial.parallel_generatedRoot_32.constraints_0.bridge h1)⟩)


theorem Stage_initial.generatedRoot.constraints_2.bridge {backend tape path w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_47_0 w_62_0 w_80_0 w_91_0 w_94_0 w_95_0 w_97_0 outputs}
    (h : Stage_initial.generatedRoot.constraints_2 backend tape path w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_47_0 w_62_0 w_80_0 w_91_0 w_94_0 w_95_0 w_97_0 outputs) : W_Stage_initial.generatedRoot.constraints_2 backend w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_47_0 w_62_0 w_80_0 w_91_0 w_94_0 w_95_0 w_97_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨h1, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h2), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h3), h4⟩)


theorem Stage_initial.generatedRoot.constraints_1.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_28_0 w_1_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_35_0 w_41_0 w_44_0 w_45_0 w_48_0 w_50_0 w_56_0 w_59_0 w_60_0 w_63_0 w_65_0 w_74_0 w_77_0 w_78_0 w_81_0 w_83_0}
    (h : Stage_initial.generatedRoot.constraints_1 backend hashModel tape path params w_0_0 w_9_0 w_28_0 w_1_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_35_0 w_41_0 w_44_0 w_45_0 w_48_0 w_50_0 w_56_0 w_59_0 w_60_0 w_63_0 w_65_0 w_74_0 w_77_0 w_78_0 w_81_0 w_83_0) : W_Stage_initial.generatedRoot.constraints_1 backend hashModel params' w_0_0 w_9_0 w_28_0 w_1_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_35_0 w_41_0 w_44_0 w_45_0 w_48_0 w_50_0 w_56_0 w_59_0 w_60_0 w_63_0 w_65_0 w_74_0 w_77_0 w_78_0 w_81_0 w_83_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨(fun i ↦ (Stage_initial.parallel_generatedRoot_24.bridge (h1 i))), h2, (fun i ↦ (Stage_initial.parallel_generatedRoot_25.bridge (h3 i))), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_26.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_27.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_29.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_30.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_31.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_32.bridge (h15 i))), h16, h17, h18, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h19), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h20), h21, h22, h23, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h24), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h25), h26, h27, h28, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h29), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h30), h31, h32⟩)


theorem Stage_initial.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0}
    (h : Stage_initial.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0) : W_Stage_initial.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h2), (MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h3), h4, (fun i ↦ (Stage_initial.parallel_generatedRoot_10.bridge (h5 i))), h6, (fun i ↦ (Stage_initial.parallel_generatedRoot_11.bridge (h7 i))), h8, (fun i ↦ (Stage_initial.parallel_generatedRoot_12.bridge (h9 i))), h10, (fun i ↦ (Stage_initial.parallel_generatedRoot_13.bridge (h11 i))), h12, (fun i ↦ (Stage_initial.parallel_generatedRoot_14.bridge (h13 i))), h14, (fun i ↦ (Stage_initial.parallel_generatedRoot_15.bridge (h15 i))), h16, (fun i ↦ (Stage_initial.parallel_generatedRoot_16.bridge (h17 i))), h18, (fun i ↦ (Stage_initial.parallel_generatedRoot_17.bridge (h19 i))), h20, (fun i ↦ (Stage_initial.parallel_generatedRoot_18.bridge (h21 i))), h22, (fun i ↦ (Stage_initial.parallel_generatedRoot_19.bridge (h23 i))), h24, (fun i ↦ (Stage_initial.parallel_generatedRoot_20.bridge (h25 i))), h26, (fun i ↦ (Stage_initial.parallel_generatedRoot_21.bridge (h27 i))), h28, (fun i ↦ (Stage_initial.parallel_generatedRoot_22.bridge (h29 i))), h30, (fun i ↦ (Stage_initial.parallel_generatedRoot_23.bridge (h31 i))), h32⟩)


theorem Stage_initial.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_initial.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_initial.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_initial.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_5_0, wit1.w_6_0, wit1.w_10_0, wit1.w_11_0, wit1.w_12_0, wit1.w_13_0, wit1.w_14_0, wit1.w_15_0, wit1.w_16_0, wit1.w_17_0, wit1.w_18_0, wit1.w_19_0, wit1.w_20_0, wit1.w_21_0, wit1.w_22_0, wit1.w_23_0, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_29_0, wit1.w_30_0, wit1.w_31_0, wit1.w_32_0, wit1.w_33_0, wit1.w_35_0, wit1.w_41_0, wit1.w_44_0, wit1.w_45_0, wit1.w_48_0, wit1.w_50_0, wit1.w_56_0, wit1.w_59_0, wit1.w_60_0, wit1.w_63_0, wit1.w_65_0, wit1.w_74_0, wit1.w_77_0, wit1.w_78_0, wit1.w_81_0, wit1.w_83_0, wit1.w_91_0, wit1.w_94_0, wit1.w_95_0⟩, (by obtain ⟨h3, h4, h5⟩ := h2; exact ⟨(Stage_initial.generatedRoot.constraints_0.bridge h3), (Stage_initial.generatedRoot.constraints_1.bridge h4), (Stage_initial.generatedRoot.constraints_2.bridge h5)⟩)⟩)
