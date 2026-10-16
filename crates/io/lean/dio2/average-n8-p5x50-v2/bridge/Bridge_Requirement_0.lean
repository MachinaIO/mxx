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


theorem Requirement_0.sequential_generatedRoot_3.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_3 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_3 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_3
  exact h


theorem Requirement_0.sequential_generatedRoot_6.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_6 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_6 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_6
  exact h


theorem Requirement_0.sequential_parallel_generatedRoot_41_11.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_parallel_generatedRoot_41_11 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_parallel_generatedRoot_41_11 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_parallel_generatedRoot_41_11
  exact h


theorem Requirement_0.parallel_generatedRoot_41.constraints_0.bridge {tape path params params' i_0 w_1_0 w_6_0 w_9_0 w_0_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 outputs}
    (h : Requirement_0.parallel_generatedRoot_41.constraints_0 tape path params i_0 w_1_0 w_6_0 w_9_0 w_0_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 outputs) : W_Requirement_0.parallel_generatedRoot_41.constraints_0 params' i_0 w_1_0 w_6_0 w_9_0 w_0_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (Requirement_0.iterRuns_mono h10 (fun _ _ _ hs13 ↦ (Requirement_0.sequential_parallel_generatedRoot_41_11.bridge hs13))), h11, h12⟩)


theorem Requirement_0.parallel_generatedRoot_41.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.parallel_generatedRoot_41 tape path params i_0 inputs outputs) : W_Requirement_0.parallel_generatedRoot_41 params' i_0 inputs outputs := by
  delta W_Requirement_0.parallel_generatedRoot_41
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0, wit1.w_10_0, wit1.w_11_0⟩, (Requirement_0.parallel_generatedRoot_41.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_generatedRoot_42.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_42 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_42 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_42
  exact h


theorem Requirement_0.sequential_sequential_generatedRoot_74_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_74_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_74_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_74_5
  exact h


theorem Requirement_0.sequential_generatedRoot_74.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_74.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_74.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_74_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_74.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_74 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_74 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_74
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_74.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_76_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_76_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_76_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_76_5
  exact h


theorem Requirement_0.sequential_generatedRoot_76.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_76.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_76.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_76_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_76.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_76 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_76 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_76
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_76.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_78_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_78_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_78_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_78_5
  exact h


theorem Requirement_0.sequential_generatedRoot_78.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_78.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_78.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_78_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_78.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_78 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_78 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_78
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_78.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_80_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_80_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_80_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_80_5
  exact h


theorem Requirement_0.sequential_generatedRoot_80.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_80.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_80.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_80_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_80.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_80 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_80 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_80
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_80.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_82_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_82_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_82_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_82_5
  exact h


theorem Requirement_0.sequential_generatedRoot_82.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_82.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_82.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_82_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_82.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_82 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_82 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_82
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_82.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_84_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_84_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_84_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_84_5
  exact h


theorem Requirement_0.sequential_generatedRoot_84.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_84.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_84.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_84_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_84.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_84 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_84 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_84
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_84.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_86_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_86_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_86_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_86_5
  exact h


theorem Requirement_0.sequential_generatedRoot_86.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_86.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_86.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_86_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_86.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_86 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_86 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_86
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_86.constraints_0.bridge h2)⟩)


theorem Requirement_0.sequential_sequential_generatedRoot_88_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_88_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_88_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_88_5
  exact h


theorem Requirement_0.sequential_generatedRoot_88.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_88.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_88.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Requirement_0.sequential_sequential_generatedRoot_88_5.bridge hs4))), h3⟩)


theorem Requirement_0.sequential_generatedRoot_88.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_88 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_88 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_88
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0⟩, (Requirement_0.sequential_generatedRoot_88.constraints_0.bridge h2)⟩)


theorem Requirement_0.generatedRoot.constraints_0.bridge {tape path params params' w_1_0 w_2_0 w_4_0 w_5_0 w_8_0 w_14_0 w_23_0 w_38_0 w_39_0 w_44_0 w_50_0 w_73_0 w_75_0 w_77_0 w_79_0 w_81_0 w_83_0 w_85_0 w_87_0 w_0_0 w_3_0 w_6_0 w_9_0 w_15_0 w_24_0 w_37_0 w_40_0 w_40_1 w_41_0 w_42_0 w_45_0 w_51_0 w_59_0 w_72_0 w_74_0 w_76_0 w_78_0 w_80_0 w_82_0 w_84_0 w_86_0 w_88_0 outputs}
    (h : Requirement_0.generatedRoot.constraints_0 tape path params w_1_0 w_2_0 w_4_0 w_5_0 w_8_0 w_14_0 w_23_0 w_38_0 w_39_0 w_44_0 w_50_0 w_73_0 w_75_0 w_77_0 w_79_0 w_81_0 w_83_0 w_85_0 w_87_0 w_0_0 w_3_0 w_6_0 w_9_0 w_15_0 w_24_0 w_37_0 w_40_0 w_40_1 w_41_0 w_42_0 w_45_0 w_51_0 w_59_0 w_72_0 w_74_0 w_76_0 w_78_0 w_80_0 w_82_0 w_84_0 w_86_0 w_88_0 outputs) : W_Requirement_0.generatedRoot.constraints_0 params' w_1_0 w_2_0 w_4_0 w_5_0 w_8_0 w_14_0 w_23_0 w_38_0 w_39_0 w_44_0 w_50_0 w_73_0 w_75_0 w_77_0 w_79_0 w_81_0 w_83_0 w_85_0 w_87_0 w_0_0 w_3_0 w_6_0 w_9_0 w_15_0 w_24_0 w_37_0 w_40_0 w_40_1 w_41_0 w_42_0 w_45_0 w_51_0 w_59_0 w_72_0 w_74_0 w_76_0 w_78_0 w_80_0 w_82_0 w_84_0 w_86_0 w_88_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (Requirement_0.iterRuns_mono h2 (fun _ _ _ hs33 ↦ (Requirement_0.sequential_generatedRoot_3.bridge hs33))), h3, (Requirement_0.iterRuns_mono h4 (fun _ _ _ hs34 ↦ (Requirement_0.sequential_generatedRoot_6.bridge hs34))), h5, h6, h7, h8, h9, (fun i ↦ (Requirement_0.parallel_generatedRoot_41.bridge (h10 i))), h11, (Requirement_0.iterRuns_mono h12 (fun _ _ _ hs35 ↦ (Requirement_0.sequential_generatedRoot_42.bridge hs35))), h13, h14, h15, h16, (Requirement_0.iterRuns_mono h17 (fun _ _ _ hs36 ↦ (Requirement_0.sequential_generatedRoot_74.bridge hs36))), h18, (Requirement_0.iterRuns_mono h19 (fun _ _ _ hs37 ↦ (Requirement_0.sequential_generatedRoot_76.bridge hs37))), h20, (Requirement_0.iterRuns_mono h21 (fun _ _ _ hs38 ↦ (Requirement_0.sequential_generatedRoot_78.bridge hs38))), h22, (Requirement_0.iterRuns_mono h23 (fun _ _ _ hs39 ↦ (Requirement_0.sequential_generatedRoot_80.bridge hs39))), h24, (Requirement_0.iterRuns_mono h25 (fun _ _ _ hs40 ↦ (Requirement_0.sequential_generatedRoot_82.bridge hs40))), h26, (Requirement_0.iterRuns_mono h27 (fun _ _ _ hs41 ↦ (Requirement_0.sequential_generatedRoot_84.bridge hs41))), h28, (Requirement_0.iterRuns_mono h29 (fun _ _ _ hs42 ↦ (Requirement_0.sequential_generatedRoot_86.bridge hs42))), h30, (Requirement_0.iterRuns_mono h31 (fun _ _ _ hs43 ↦ (Requirement_0.sequential_generatedRoot_88.bridge hs43))), h32⟩)


theorem Requirement_0.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Requirement_0.generatedRoot tape path params inputs outputs) : W_Requirement_0.generatedRoot params' inputs outputs := by
  delta W_Requirement_0.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_6_0, wit1.w_9_0, wit1.w_15_0, wit1.w_24_0, wit1.w_40_0, wit1.w_40_1, wit1.w_41_0, wit1.w_42_0, wit1.w_45_0, wit1.w_51_0, wit1.w_59_0, wit1.w_74_0, wit1.w_76_0, wit1.w_78_0, wit1.w_80_0, wit1.w_82_0, wit1.w_84_0, wit1.w_86_0, wit1.w_88_0⟩, (Requirement_0.generatedRoot.constraints_0.bridge h2)⟩)
