import Stage_payload
import W_Stage_payload
import ExponentSampling

/-! Every run of the average-case relations of `Stage_payload` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_payload.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_payload.generatedRoot.constraints_0.bridge {tape path w_0_0 outputs}
    (h : Stage_payload.generatedRoot.constraints_0 tape path w_0_0 outputs) : W_Stage_payload.generatedRoot.constraints_0 w_0_0 outputs := by
  exact (by obtain ⟨h1, h2⟩ := h; exact ⟨(MxxExponentLut.uniformIntervalSampleAt_sample (by decide) h1), h2⟩)


theorem Stage_payload.generatedRoot.bridge {tape path params params' x3 outputs}
    (h : Stage_payload.generatedRoot tape path params x3 outputs) : W_Stage_payload.generatedRoot params' x3 outputs := by
  delta W_Stage_payload.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_0_0⟩, (Stage_payload.generatedRoot.constraints_0.bridge h2)⟩)
