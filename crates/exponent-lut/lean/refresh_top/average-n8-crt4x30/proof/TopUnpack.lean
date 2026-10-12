import W_Stage_online_eval
import TopSetup

/-! The members of the online evaluation's residue family that the decode stage reads. -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
/-- The online evaluation's unpacked residues are the members of its residue family. -/
theorem online_eval_unpack {model : HashModel} {params : W_Stage_online_eval.Params} {inputs outputs}
    (h : W_Stage_online_eval.generatedRoot Backend.backend model params inputs outputs) :
    outputs.2.2.2.2.2.2.2.1 = outputs.2.2.2.2.2.2.1 ⟨0, by decide⟩ ∧
      outputs.2.2.2.2.2.2.2.2.1 = outputs.2.2.2.2.2.2.1 ⟨1, by decide⟩ ∧
      outputs.2.2.2.2.2.2.2.2.2.1 = outputs.2.2.2.2.2.2.1 ⟨2, by decide⟩ ∧
      outputs.2.2.2.2.2.2.2.2.2.2.1 = outputs.2.2.2.2.2.2.1 ⟨3, by decide⟩ ∧
      outputs.2.2.2.2.2.2.2.2.2.2.2.1 = outputs.2.2.2.2.2.2.1 ⟨4, by decide⟩ ∧
      outputs.2.2.2.2.2.2.2.2.2.2.2.2.1 = outputs.2.2.2.2.2.2.1 ⟨5, by decide⟩ ∧
      outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = outputs.2.2.2.2.2.2.1 ⟨6, by decide⟩ ∧
      outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = outputs.2.2.2.2.2.2.1 ⟨7, by decide⟩ := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval.generatedRoot.body at h
  simp only [W_Stage_online_eval.generatedRoot.constraints_0] at h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, u0, u1, u2, u3, u4, u5, u6, u7, hout⟩ := h
  subst hout
  exact ⟨u0, u1, u2, u3, u4, u5, u6, u7⟩

end RefreshTopProof
