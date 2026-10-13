import Stage_mask
import W_Stage_mask
import ExponentSampling

/-! Every run of the average-case relations of `Stage_mask` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_mask.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_mask.generatedRoot.constraints_0.bridge {tape path w_0_0 outputs}
    (h : Stage_mask.generatedRoot.constraints_0 tape path w_0_0 outputs) : W_Stage_mask.generatedRoot.constraints_0 w_0_0 outputs := by
  obtain ⟨a1, a2⟩ := h
  exact ⟨(MxxExponentLut.uniformIntervalSampleAt_sample (by decide) a1), a2⟩


theorem Stage_mask.generatedRoot.bridge {tape path params params' x3 outputs}
    (h : Stage_mask.generatedRoot tape path params x3 outputs) : W_Stage_mask.generatedRoot params' x3 outputs := by
  delta W_Stage_mask.generatedRoot
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_0_0⟩, (Stage_mask.generatedRoot.constraints_0.bridge hb)⟩
