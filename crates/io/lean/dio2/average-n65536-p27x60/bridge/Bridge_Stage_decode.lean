import Stage_decode
import W_Stage_decode
import ExponentSampling

/-! Every run of the average-case relations of `Stage_decode` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_decode.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_decode.sequential_generatedRoot_3.bridge {tape path params params' i_0 inputs outputs}
    (h : Stage_decode.sequential_generatedRoot_3 tape path params i_0 inputs outputs) : W_Stage_decode.sequential_generatedRoot_3 params' i_0 inputs outputs := by
  delta W_Stage_decode.sequential_generatedRoot_3
  exact h


theorem Stage_decode.generatedRoot.constraints_0.bridge {tape path params params' w_1_0 w_0_0 w_2_0 w_3_0 w_4_0 w_5_0 outputs}
    (h : Stage_decode.generatedRoot.constraints_0 tape path params w_1_0 w_0_0 w_2_0 w_3_0 w_4_0 w_5_0 outputs) : W_Stage_decode.generatedRoot.constraints_0 params' w_1_0 w_0_0 w_2_0 w_3_0 w_4_0 w_5_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5⟩ := h; exact ⟨h1, h2, (Stage_decode.iterRuns_mono h3 (fun _ _ _ hs6 ↦ (Stage_decode.sequential_generatedRoot_3.bridge hs6))), h4, h5⟩)


theorem Stage_decode.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Stage_decode.generatedRoot tape path params inputs outputs) : W_Stage_decode.generatedRoot params' inputs outputs := by
  delta W_Stage_decode.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_3_0⟩, (Stage_decode.generatedRoot.constraints_0.bridge h2)⟩)
