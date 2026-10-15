import Requirement_0
import W_Requirement_0
import ExponentSampling

/-! Every run of the average-case relations of `Requirement_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Requirement_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Requirement_0.sequential_sequential_generatedRoot_298_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_298_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_298_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_298_5
  exact h


theorem Requirement_0.sequential_generatedRoot_298.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_298.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_298.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_298_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_298.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_298 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_298 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_298
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_298.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_300_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_300_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_300_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_300_5
  exact h


theorem Requirement_0.sequential_generatedRoot_300.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_300.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_300.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_300_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_300.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_300 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_300 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_300
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_300.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_302_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_302_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_302_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_302_5
  exact h


theorem Requirement_0.sequential_generatedRoot_302.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_302.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_302.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_302_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_302.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_302 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_302 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_302
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_302.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_304_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_304_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_304_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_304_5
  exact h


theorem Requirement_0.sequential_generatedRoot_304.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_304.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_304.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_304_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_304.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_304 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_304 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_304
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_304.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_306_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_306_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_306_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_306_5
  exact h


theorem Requirement_0.sequential_generatedRoot_306.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_306.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_306.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_306_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_306.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_306 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_306 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_306
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_306.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_308_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_308_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_308_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_308_5
  exact h


theorem Requirement_0.sequential_generatedRoot_308.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_308.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_308.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_308_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_308.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_308 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_308 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_308
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_308.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_310_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_310_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_310_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_310_5
  exact h


theorem Requirement_0.sequential_generatedRoot_310.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_310.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_310.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_310_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_310.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_310 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_310 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_310
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_310.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_312_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_312_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_312_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_312_5
  exact h


theorem Requirement_0.sequential_generatedRoot_312.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_312.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_312.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_312_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_312.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_312 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_312 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_312
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_312.constraints_0.bridge h2)⟩)


theorem Requirement_0.generatedRoot.constraints_1.bridge {tape path params params' w_301_0 w_303_0 w_305_0 w_307_0 w_309_0 w_311_0 w_300_0 w_302_0 w_304_0 w_306_0 w_308_0 w_310_0 w_312_0 outputs}
    (h : Requirement_0.generatedRoot.constraints_1 tape path params w_301_0 w_303_0 w_305_0 w_307_0 w_309_0 w_311_0 w_300_0 w_302_0 w_304_0 w_306_0 w_308_0 w_310_0 w_312_0 outputs) : W_Requirement_0.generatedRoot.constraints_1 params' w_301_0 w_303_0 w_305_0 w_307_0 w_309_0 w_311_0 w_300_0 w_302_0 w_304_0 w_306_0 w_308_0 w_310_0 w_312_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs14 ↦ (Requirement_0.sequential_generatedRoot_302.bridge hs14))), h3, (Requirement_0.iterRuns_mono h4 (fun _ _ _ hs15 ↦ (Requirement_0.sequential_generatedRoot_304.bridge hs15))), h5, (Requirement_0.iterRuns_mono h6 (fun _ _ _ hs16 ↦ (Requirement_0.sequential_generatedRoot_306.bridge hs16))), h7, (Requirement_0.iterRuns_mono h8 (fun _ _ _ hs17 ↦ (Requirement_0.sequential_generatedRoot_308.bridge hs17))), h9, (Requirement_0.iterRuns_mono h10 (fun _ _ _ hs18 ↦ (Requirement_0.sequential_generatedRoot_310.bridge hs18))), h11, (Requirement_0.iterRuns_mono h12 (fun _ _ _ hs19 ↦ (Requirement_0.sequential_generatedRoot_312.bridge hs19))), h13⟩)


theorem Requirement_0.generatedRoot.constraints_0.bridge {tape path params params' w_2_0 w_8_0 w_17_0 w_61_0 w_67_0 w_175_0 w_181_0 w_190_0 w_206_0 w_212_0 w_268_0 w_274_0 w_297_0 w_299_0 w_3_0 w_9_0 w_18_0 w_33_0 w_38_0 w_62_0 w_68_0 w_76_0 w_91_0 w_96_0 w_104_0 w_119_0 w_124_0 w_132_0 w_147_0 w_152_0 w_160_0 w_176_0 w_182_0 w_191_0 w_207_0 w_213_0 w_224_0 w_240_0 w_245_0 w_269_0 w_275_0 w_283_0 w_296_0 w_298_0 w_300_0}
    (h : Requirement_0.generatedRoot.constraints_0 tape path params w_2_0 w_8_0 w_17_0 w_61_0 w_67_0 w_175_0 w_181_0 w_190_0 w_206_0 w_212_0 w_268_0 w_274_0 w_297_0 w_299_0 w_3_0 w_9_0 w_18_0 w_33_0 w_38_0 w_62_0 w_68_0 w_76_0 w_91_0 w_96_0 w_104_0 w_119_0 w_124_0 w_132_0 w_147_0 w_152_0 w_160_0 w_176_0 w_182_0 w_191_0 w_207_0 w_213_0 w_224_0 w_240_0 w_245_0 w_269_0 w_275_0 w_283_0 w_296_0 w_298_0 w_300_0) : W_Requirement_0.generatedRoot.constraints_0 params' w_2_0 w_8_0 w_17_0 w_61_0 w_67_0 w_175_0 w_181_0 w_190_0 w_206_0 w_212_0 w_268_0 w_274_0 w_297_0 w_299_0 w_3_0 w_9_0 w_18_0 w_33_0 w_38_0 w_62_0 w_68_0 w_76_0 w_91_0 w_96_0 w_104_0 w_119_0 w_124_0 w_132_0 w_147_0 w_152_0 w_160_0 w_176_0 w_182_0 w_191_0 w_207_0 w_213_0 w_224_0 w_240_0 w_245_0 w_269_0 w_275_0 w_283_0 w_296_0 w_298_0 w_300_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, (Requirement_0.iterRuns_mono h30 (fun _ _ _ hs33 ↦ (Requirement_0.sequential_generatedRoot_298.bridge hs33))), h31, (Requirement_0.iterRuns_mono h32 (fun _ _ _ hs34 ↦ (Requirement_0.sequential_generatedRoot_300.bridge hs34)))⟩)


theorem Requirement_0.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Requirement_0.generatedRoot tape path params inputs outputs) : W_Requirement_0.generatedRoot params' inputs outputs := by
  delta W_Requirement_0.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_9_0, wit1.w_18_0, wit1.w_33_0, wit1.w_38_0, wit1.w_62_0, wit1.w_68_0, wit1.w_76_0, wit1.w_91_0, wit1.w_96_0, wit1.w_104_0, wit1.w_119_0, wit1.w_124_0, wit1.w_132_0, wit1.w_147_0, wit1.w_152_0, wit1.w_160_0, wit1.w_176_0, wit1.w_182_0, wit1.w_191_0, wit1.w_207_0, wit1.w_213_0, wit1.w_240_0, wit1.w_245_0, wit1.w_269_0, wit1.w_275_0, wit1.w_283_0, wit1.w_298_0, wit1.w_300_0, wit1.w_302_0, wit1.w_304_0, wit1.w_306_0, wit1.w_308_0, wit1.w_310_0, wit1.w_312_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Requirement_0.generatedRoot.constraints_0.bridge h3), (Requirement_0.generatedRoot.constraints_1.bridge h4)⟩)⟩)
