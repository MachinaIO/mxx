import Bridge_Ideal
import Bridge_Requirement_0
import Bridge_Stage_anchor_publics
import Bridge_Stage_anchors
import Bridge_Stage_decode
import Bridge_Stage_import_0
import Bridge_Stage_import_1
import Bridge_Stage_initial
import Bridge_Stage_keys_import_0_0
import Bridge_Stage_keys_import_1_1
import Bridge_Stage_keys_online_0_0
import Bridge_Stage_keys_online_1_1
import Bridge_Stage_mask
import Bridge_Stage_online_0
import Bridge_Stage_online_1
import Bridge_Stage_payload
import Bridge_Stage_preimages_0_0
import Bridge_Stage_preimages_1_1
import Bridge_Stage_preprocess_0_0
import Bridge_Stage_preprocess_1_1
import Bridge_Stage_zero
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
  obtain ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22⟩ := h
  exact ⟨r0, Stage_payload.generatedRoot.bridge r1, Stage_anchors.generatedRoot.bridge r2, Stage_anchor_publics.generatedRoot.bridge r3, Stage_mask.generatedRoot.bridge r4, Stage_initial.generatedRoot.bridge r5, Stage_zero.generatedRoot.bridge r6, Stage_preprocess_0_0.generatedRoot.bridge r7, Stage_keys_import_0_0.generatedRoot.bridge r8, Stage_keys_online_0_0.generatedRoot.bridge r9, Stage_preimages_0_0.generatedRoot.bridge r10, Stage_import_0.generatedRoot.bridge r11, Stage_online_0.generatedRoot.bridge r12, Stage_preprocess_1_1.generatedRoot.bridge r13, Stage_keys_import_1_1.generatedRoot.bridge r14, Stage_keys_online_1_1.generatedRoot.bridge r15, Stage_preimages_1_1.generatedRoot.bridge r16, Stage_import_1.generatedRoot.bridge r17, Stage_online_1.generatedRoot.bridge r18, Stage_decode.generatedRoot.bridge r19, Requirement_0.generatedRoot.bridge r20, Ideal.generatedRoot.bridge r21, r22⟩
