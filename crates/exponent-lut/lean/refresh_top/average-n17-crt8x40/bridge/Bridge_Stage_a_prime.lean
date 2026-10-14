import Stage_a_prime
import W_Stage_a_prime
import ExponentSampling

/-! Every run of the average-case relations of `Stage_a_prime` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_a_prime.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_a_prime.generatedRoot.bridge {hashModel tape path params params' inputs outputs}
    (h : Stage_a_prime.generatedRoot hashModel tape path params inputs outputs) : W_Stage_a_prime.generatedRoot hashModel params' inputs outputs := by
  delta W_Stage_a_prime.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_3_0⟩, h2⟩)
