import Bridge_Ideal
import Bridge_Requirement_0
import Bridge_Stage_decode
import Bridge_Stage_encrypt_0_0
import Bridge_Stage_encrypt_0_1
import Bridge_Stage_mask
import Bridge_Stage_online_eval_0_0
import Bridge_Stage_online_eval_0_1
import Bridge_Stage_payload
import Bridge_Stage_preprocess_0_0
import Bridge_Stage_preprocess_0_1
import W_Claim

/-! Every average-case run is a worst-case run (written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem GeneratedClaim.runs_bridge {hashModel : MxxRuntime.HashModel}
    {external : GeneratedClaim.ExternalInputs} {tape : MxxRuntime.SampleTape}
    {execution : GeneratedClaim.Execution}
    (h : GeneratedClaim.Runs hashModel external tape execution) :
    W_GeneratedClaim.Runs hashModel external execution := by
  delta W_GeneratedClaim.Runs
  obtain ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12⟩ := h
  exact ⟨r0, Stage_payload.generatedRoot.bridge r1, Stage_mask.generatedRoot.bridge r2, Stage_preprocess_0_0.generatedRoot.bridge r3, Stage_encrypt_0_0.generatedRoot.bridge r4, Stage_online_eval_0_0.generatedRoot.bridge r5, Stage_preprocess_0_1.generatedRoot.bridge r6, Stage_encrypt_0_1.generatedRoot.bridge r7, Stage_online_eval_0_1.generatedRoot.bridge r8, Stage_decode.generatedRoot.bridge r9, Requirement_0.generatedRoot.bridge r10, Ideal.generatedRoot.bridge r11, r12⟩
