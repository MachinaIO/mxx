import Stage_keys_import_1_1
import W_Stage_keys_import_1_1
import ExponentSampling

/-! Every run of the average-case relations of `Stage_keys_import_1_1` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_keys_import_1_1.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_keys_import_1_1.parallel_generatedRoot_6.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_6 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_6 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_6
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_11.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_11 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_11 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_11
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_14.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_14 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_14 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_14
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_15.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_15 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_15 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_15
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_16.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_16 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_16 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_16
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_17.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_17 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_17 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_17
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_18.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_18 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_18 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_18
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_19.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_19 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_19 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_19
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_20.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_20 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_20 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_20
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_21.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_21 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_21 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_21
  exact h


theorem Stage_keys_import_1_1.parallel_generatedRoot_22.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_1_1.parallel_generatedRoot_22 tape path params x3 inputs outputs) : W_Stage_keys_import_1_1.parallel_generatedRoot_22 params' x3 inputs outputs := by
  delta W_Stage_keys_import_1_1.parallel_generatedRoot_22
  exact h


theorem Stage_keys_import_1_1.generatedRoot.constraints_0.bridge {tape path params params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_14_0 w_14_1 w_15_0 w_15_1 w_16_0 w_16_1 w_17_0 w_17_1 w_18_0 w_18_1 w_19_0 w_19_1 w_20_0 w_20_1 w_21_0 w_21_1 w_22_0 w_22_1 outputs}
    (h : Stage_keys_import_1_1.generatedRoot.constraints_0 tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_14_0 w_14_1 w_15_0 w_15_1 w_16_0 w_16_1 w_17_0 w_17_1 w_18_0 w_18_1 w_19_0 w_19_1 w_20_0 w_20_1 w_21_0 w_21_1 w_22_0 w_22_1 outputs) : W_Stage_keys_import_1_1.generatedRoot.constraints_0 params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_14_0 w_14_1 w_15_0 w_15_1 w_16_0 w_16_1 w_17_0 w_17_1 w_18_0 w_18_1 w_19_0 w_19_1 w_20_0 w_20_1 w_21_0 w_21_1 w_22_0 w_22_1 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23⟩ := h; exact ⟨h1, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_6.bridge (h2 i))), h3, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_11.bridge (h4 i))), h5, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_14.bridge (h6 i))), h7, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_15.bridge (h8 i))), h9, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_16.bridge (h10 i))), h11, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_17.bridge (h12 i))), h13, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_18.bridge (h14 i))), h15, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_19.bridge (h16 i))), h17, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_20.bridge (h18 i))), h19, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_21.bridge (h20 i))), h21, (fun i ↦ (Stage_keys_import_1_1.parallel_generatedRoot_22.bridge (h22 i))), h23⟩)


theorem Stage_keys_import_1_1.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Stage_keys_import_1_1.generatedRoot tape path params inputs outputs) : W_Stage_keys_import_1_1.generatedRoot params' inputs outputs := by
  delta W_Stage_keys_import_1_1.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_11_0, wit1.w_11_1, wit1.w_14_0, wit1.w_14_1, wit1.w_15_0, wit1.w_15_1, wit1.w_16_0, wit1.w_16_1, wit1.w_17_0, wit1.w_17_1, wit1.w_18_0, wit1.w_18_1, wit1.w_19_0, wit1.w_19_1, wit1.w_20_0, wit1.w_20_1, wit1.w_21_0, wit1.w_21_1, wit1.w_22_0, wit1.w_22_1⟩, (Stage_keys_import_1_1.generatedRoot.constraints_0.bridge h2)⟩)
