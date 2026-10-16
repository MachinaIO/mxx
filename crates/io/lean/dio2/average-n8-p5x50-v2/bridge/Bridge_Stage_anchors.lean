import Stage_anchors
import W_Stage_anchors
import ExponentSampling

/-! Every run of the average-case relations of `Stage_anchors` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_anchors.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_anchors.generatedRoot.constraints_0.bridge {backend tape path w_0_1 w_0_0 w_1_1 w_1_0 outputs}
    (h : Stage_anchors.generatedRoot.constraints_0 backend tape path w_0_1 w_0_0 w_1_1 w_1_0 outputs) : W_Stage_anchors.generatedRoot.constraints_0 backend w_0_1 w_0_0 w_1_1 w_1_0 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨(MxxRuntime.trapdoorSampleAt_sample h1), (MxxRuntime.trapdoorSampleAt_sample h2), h3⟩)


theorem Stage_anchors.generatedRoot.bridge {backend tape path params params' x4 outputs}
    (h : Stage_anchors.generatedRoot backend tape path params x4 outputs) : W_Stage_anchors.generatedRoot backend params' x4 outputs := by
  delta W_Stage_anchors.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_0_1, wit1.w_0_0, wit1.w_1_1, wit1.w_1_0⟩, (Stage_anchors.generatedRoot.constraints_0.bridge h2)⟩)
