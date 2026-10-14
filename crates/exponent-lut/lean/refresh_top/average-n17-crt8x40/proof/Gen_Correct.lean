import W_Claim
import Gen_Requirement
import TopDecodeLoops
import Gen_Part

/-! Every run of the generated claim decodes the ideal output (written by
`generator/gen_correct2.py`): the secrets are ternary, the requirement gives its facts
(`GenRequirement.requirement_facts`), the part theorem gives every CRT term the decode stage reads
(`∀ i`, one application per instance index), the looped decode outputs `0`
(`DecodeLoops.decode_zero_loops`), and the ideal output is `0`. -/

set_option maxRecDepth 65536

namespace RefreshTopProof.GenCorrect

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof GenDecode

/-- The secret of `W_Stage_payload` is one ternary polynomial. -/
theorem secret_Stage_payload {out : ExactMatrix W_Stage_payload.modulus_0 131072 1 1}
    (h : W_Stage_payload.generatedRoot W_GeneratedClaim.stage_0_params () out) :
    ∃ s : ErrorPoly 131072, out = (fun _ _ ↦ reducePoly W_Stage_payload.modulus_0 131072 s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

/-- The secret of `W_Stage_mask` is one ternary polynomial. -/
theorem secret_Stage_mask {out : ExactMatrix W_Stage_mask.modulus_0 131072 1 1}
    (h : W_Stage_mask.generatedRoot W_GeneratedClaim.stage_1_params () out) :
    ∃ s : ErrorPoly 131072, out = (fun _ _ ↦ reducePoly W_Stage_mask.modulus_0 131072 s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

/-- The ideal output is `0`. -/
theorem ideal_zero {e out : Int} (h : W_Ideal.generatedRoot W_GeneratedClaim.ideal_params e out) :
    out = 0 := by
  obtain ⟨_, h⟩ := h
  simp only [W_Ideal.generatedRoot.body, W_Ideal.generatedRoot.constraints_0] at h
  rw [h, mul_zero]

/-- The decode stage's inputs in a run. -/
abbrev decodeInputs (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution) :
    DecodeLoops.Inputs :=
  (execution.«stage_8».2.2.2.2.2.2.1, external.input_4, execution.«stage_2», execution.«stage_0», external.input_2, external.input_3, ())

/-- The backend's gadget matrix of the decode stage's gadget modulus. -/
theorem decode_gadget : ∃ G : ExactMatrix 1942566534251559791942281995946931586214205534210157066971588917963081260105189818369 131072 1 28, gadgetMatrixRuns Backend.backend 1024 28 G :=
  ⟨_, Backend.layout4, by simp [Backend.backend], by decide, by decide, by decide, rfl⟩

/-- The online CRT tables are the decode tables: instance `i` uses prime and basis element `i / R`
(one `decide` over the instance tables). -/
theorem tables_q : ∀ i : Fin GenPart.S, GenPart.qOf i = DecodeLoops.decode_qN (i / GenPart.R) := by decide

theorem tables_K : ∀ i : Fin GenPart.S, GenPart.KOf i = DecodeLoops.decode_KN (i / GenPart.R) := by decide

/-- The part theorem's requirement hypothesis `GenPart.Req` is exactly the requirement facts. -/
theorem req_of_facts {external : GeneratedClaim.ExternalInputs} (hreq : GenRequirement.Facts (external.input_7) (external.input_5) (external.input_6) (external.input_3) (external.input_1) (external.input_13) (external.input_18) (external.input_19) (external.input_20) (external.input_21) (external.input_22) (external.input_23) (external.input_24) (external.input_25) (external.input_26) (external.input_27) (external.input_28) (external.input_29) (external.input_30) (external.input_31) (external.input_32) (external.input_33) (external.input_34) (external.input_35) (external.input_36) (external.input_37) (external.input_38) (external.input_39) (external.input_40) (external.input_41) (external.input_42) (external.input_43) (external.input_44) (external.input_45) (external.input_46) (external.input_47) (external.input_48) (external.input_49) (external.input_50) (external.input_51) (external.input_52) (external.input_53) (external.input_54) (external.input_55) (external.input_56) (external.input_57) (external.input_58) (external.input_59) (external.input_60) (external.input_61) (external.input_62) (external.input_63) (external.input_64) (external.input_65) (external.input_66) (external.input_67) (external.input_68) (external.input_69) (external.input_70) (external.input_71) (external.input_72) (external.input_73) (external.input_74) (external.input_75) (external.input_76) (external.input_77) (external.input_78) (external.input_79) (external.input_80) (external.input_81) (external.input_82) (external.input_83) (external.input_84) (external.input_85) (external.input_86) (external.input_87) (external.input_88) (external.input_89) (external.input_90) (external.input_91) (external.input_92) (external.input_93) (external.input_94) (external.input_95) (external.input_96) (external.input_97) (external.input_98) (external.input_99) (external.input_100) (external.input_101) (external.input_102) (external.input_103) (external.input_104) (external.input_105) (external.input_106) (external.input_107) (external.input_108) (external.input_109) (external.input_110) (external.input_111) (external.input_112) (external.input_113) (external.input_114) (external.input_115) (external.input_116) (external.input_117) (external.input_118) (external.input_119) (external.input_120) (external.input_121) (external.input_122) (external.input_123) (external.input_125) (external.input_126) (external.input_127) (external.input_128) (external.input_129) (external.input_130) (external.input_131) (external.input_132) (external.input_133) (external.input_134) (external.input_135) (external.input_136) (external.input_137) (external.input_138) (external.input_139) (external.input_140) (external.input_141) (external.input_142) (external.input_143) (external.input_144) (external.input_145) (external.input_146) (external.input_147) (external.input_148) (external.input_149) (external.input_150) (external.input_151) (external.input_152) (external.input_153) (external.input_154) (external.input_155) (external.input_156) (external.input_157) (external.input_158) (external.input_159) (external.input_160) (external.input_161) (external.input_162) (external.input_163) (external.input_164) (external.input_165) (external.input_166) (external.input_167) (external.input_168) (external.input_169) (external.input_170) (external.input_171) (external.input_172) (external.input_173) (external.input_174) (external.input_175) (external.input_176) (external.input_177) (external.input_178) (external.input_179) (external.input_180) (external.input_181) (external.input_182) (external.input_183) (external.input_184) (external.input_185) (external.input_186) (external.input_187) (external.input_188) (external.input_189) (external.input_190) (external.input_191) (external.input_192) (external.input_4)) :
    GenPart.Req external := hreq

set_option maxHeartbeats 0 in
/-- Every run decodes the ideal output. -/
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    execution.«stage_9» = execution.«ideal» := by
  have hrun := h
  obtain ⟨hv, r_Stage_payload, r_Stage_mask, r_Stage_a_prime, r_Stage_sigma_preprocess, r_Stage_sigma_encrypt, r_Stage_sigma_online_eval, r_Stage_preprocess, r_Stage_encrypt, r_Stage_online_eval, r_Stage_decode, r_Requirement_0, r_Ideal, hreq⟩ := h
  have hsec_Stage_payload := secret_Stage_payload r_Stage_payload
  have hsec_Stage_mask := secret_Stage_mask r_Stage_mask
  have F := GenRequirement.requirement_facts r_Requirement_0 hreq
  have hreq := req_of_facts F
  obtain ⟨G, hG⟩ := decode_gadget
  rw [ideal_zero r_Ideal]
  refine DecodeLoops.decode_zero_loops r_Stage_decode hG (GenPart.errE external) (GenPart.errE_bound hreq) (fun i ↦ ?_)
  have hT := GenPart.part hrun hreq i hG
  rw [tables_q i, tables_K i] at hT
  exact hT

end RefreshTopProof.GenCorrect
