import Stage_preimages_0_0
import W_Stage_preimages_0_0
import ExponentSampling

/-! Every run of the average-case relations of `Stage_preimages_0_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_preimages_0_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_preimages_0_0.parallel_generatedRoot_3.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_3 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_3 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_3
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_5.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_5 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_5 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_5
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_7.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_7 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_7 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_7
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_9.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_9 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_9 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_9
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_11.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_11 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_11 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_11
  exact h


theorem Stage_preimages_0_0.generatedRoot.constraints_0.bridge {backend tape path params params' w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 outputs}
    (h : Stage_preimages_0_0.generatedRoot.constraints_0 backend tape path params w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 outputs) : W_Stage_preimages_0_0.generatedRoot.constraints_0 backend params' w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_3.bridge (h2 i))), h3, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_5.bridge (h4 i))), h5, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_7.bridge (h6 i))), h7, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_9.bridge (h8 i))), h9, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_11.bridge (h10 i))), h11⟩)


theorem Stage_preimages_0_0.generatedRoot.bridge {backend tape path params params' inputs outputs}
    (h : Stage_preimages_0_0.generatedRoot backend tape path params inputs outputs) : W_Stage_preimages_0_0.generatedRoot backend params' inputs outputs := by
  delta W_Stage_preimages_0_0.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_5_0, wit1.w_7_0, wit1.w_9_0, wit1.w_11_0⟩, (Stage_preimages_0_0.generatedRoot.constraints_0.bridge h2)⟩)
