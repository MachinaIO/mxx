import Stage_anchor_publics
import W_Stage_anchor_publics
import ExponentSampling

/-! Every run of the average-case relations of `Stage_anchor_publics` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_anchor_publics.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_anchor_publics.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Stage_anchor_publics.generatedRoot tape path params inputs outputs) : W_Stage_anchor_publics.generatedRoot params' inputs outputs := by
  delta W_Stage_anchor_publics.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨⟩, h2⟩)
