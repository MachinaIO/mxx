import Claim
import TopSetup
import ExponentRuntime

/-!
# The secrets and the ideal output

The payload `r` and the mask `s` are ternary samples at `Q₀`; the ideal output is `0`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem payload_ternary {out : ExactMatrix Q0 N 1 1}
    (h : Stage_payload.generatedRoot GeneratedClaim.stage_0_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q0 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

theorem mask_ternary {out : ExactMatrix Q0 N 1 1}
    (h : Stage_mask.generatedRoot GeneratedClaim.stage_1_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q0 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

theorem ideal_zero {e : Int} {out : Int} (h : Ideal.generatedRoot GeneratedClaim.ideal_params e out) :
    out = 0 := by
  obtain ⟨w, h⟩ := h
  simp only [Ideal.generatedRoot.body, Ideal.generatedRoot.constraints_0] at h
  rw [h, mul_zero]

end RefreshTopProof
