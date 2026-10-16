import Ideal
import W_Ideal
import ExponentSampling

/-! Every run of the average-case relations of `Ideal` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Ideal.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Ideal.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Ideal.generatedRoot tape path params inputs outputs) : W_Ideal.generatedRoot params' inputs outputs := by
  delta W_Ideal.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_20_0, wit1.w_23_0, wit1.w_39_0⟩, h2⟩)
