import Stage_zero
import W_Stage_zero
import ExponentSampling

/-! Every run of the average-case relations of `Stage_zero` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_zero.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_zero.generatedRoot.bridge {tape path params params' x3 outputs}
    (h : Stage_zero.generatedRoot tape path params x3 outputs) : W_Stage_zero.generatedRoot params' x3 outputs := by
  delta W_Stage_zero.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_4_0, wit1.w_6_0, wit1.w_8_0, wit1.w_10_0, wit1.w_12_0, wit1.w_14_0, wit1.w_16_0, wit1.w_18_0, wit1.w_20_0, wit1.w_22_0, wit1.w_24_0, wit1.w_26_0, wit1.w_28_0, wit1.w_37_0, wit1.w_46_0, wit1.w_51_0, wit1.w_56_0, wit1.w_59_0, wit1.w_62_0, wit1.w_67_0, wit1.w_72_0⟩, h2⟩)
