import PeChainAll
import PeKeyLink
import PeDecodeLink
import Claim

/-!
# The certificate

Every run decrypts to the ideal message: the requirement fixes a chain ending at `0`, the online
evaluation encodes `0` under the preprocessed output key, and the key of the predicate turns the
phase into `⌊Q/2⌋ m` plus an error below `Q / 4`. The bad set is empty, so its measure is `0`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

theorem payload_explicit {tape : SampleTape} {out : ExactMatrix Q N 1 1}
    (h : Stage_payload.generatedRoot tape [2] GeneratedClaim.stage_2_params () out) :
    ∃ sm : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q N sm) ∧ polyNorm sm ≤ 1 := by
  obtain ⟨w, h0, hout⟩ := h
  obtain ⟨hx, hb⟩ := ternarySampleAt_explicit hN h0
  subst hout
  exact ⟨_, hx, polyNorm_le_of_coeff fun k ↦ by have := hb k; rw [abs_le] at this; omega⟩

theorem message_one : (((1 : Nat) : Int) : ErrorPoly N) * message 1 0 = (((1 : Nat) : Int) : ErrorPoly N) * Xe 0 := by
  simp [message, Xe]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
theorem run_correct {model : HashModel} {external : GeneratedClaim.ExternalInputs} {ω : SampleTape}
    {exec : GeneratedClaim.Execution} (hruns : GeneratedClaim.Runs model external ω exec) :
    exec.«stage_9» = exec.«ideal» := by
  obtain ⟨st0, st1, st2, st3, st4, st5, st6, st7, st8, st9, rq, idl⟩ := exec
  obtain ⟨hvalid, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, hreq, hideal, hrq⟩ := hruns
  dsimp only at h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 hreq hideal hrq ⊢
  obtain ⟨-, hxb, hmb, hc1, he0, -, htb⟩ := hvalid
  have hc : external.input_3 = 1 := by have := hc1; omega
  have he : external.input_4 = 0 := by have := he0; omega
  -- Setup and payload.
  obtain ⟨hB0, hkind⟩ := setup_trap h0
  obtain ⟨r, hr, hBQ⟩ := setup_public h1
  obtain ⟨sm, hsec, hs⟩ := payload_explicit h2
  -- Prelude and encryption.
  obtain ⟨hattr, hbQ, heb, hcm, hem⟩ := prelude_explicit h3 hsec hs hxb
  rw [hc, he] at h4
  obtain ⟨hhelpE, hcompE, hrhsE, hstate, hstateB⟩ := encrypt_explicit h4 hsec hs (prelude_helpers h3 hsec)
  obtain ⟨hcopy0, hcopy1⟩ := encrypt_helpers h4
  rw [← hcopy0, ← hcopy1] at hcompE
  -- Preprocessing.
  obtain ⟨pw, pbody⟩ := h5
  let pctx : Stage_preprocess.generatedRoot.Ctx Backend.backend model ω [5]
      GeneratedClaim.stage_5_params := ⟨(external.input_5, external.input_0, external.input_6, ()), pw⟩
  have G := Stage_preprocess.generatedRoot.facts pctx pbody
  have hout5 := G.c9.hout
  subst hout5
  -- Online evaluation.
  obtain ⟨ow, obody⟩ := h8
  have F := Stage_online_eval.generatedRoot.facts ⟨_, ow⟩ obody
  have hout7 := F.c5.hout
  subst hout7
  -- The requirement.
  obtain ⟨hx8, ht8, ht2, v, hv0, hv4, hchain⟩ := requirement_facts hreq hrq
  -- The chain.
  have hout := chain_all F G rfl rfl rfl rfl rfl rfl
    rfl rfl rfl rfl
    (leafFamilies_eta (fun slot ↦ preLeafT_0 pctx slot 0))
    (leafFamilies_eta (fun slot ↦ preLeafT_0 pctx slot 1))
    rfl rfl rfl rfl
    (leafFamilies_eta (fun slot ↦ preLeafT_1 pctx slot 0))
    (leafFamilies_eta (fun slot ↦ preLeafT_1 pctx slot 1))
    hs
    (eB := fun i j ↦ attrErr ω [3] i j) (ez0 := fun i ↦ rhsCompErr ω [4] i) (ez := fun i ↦ compErr ω [4] i)
    ⟨stateErr ω [4], hstateB, fun j ↦ by have := hstate j; rw [message_one] at this; exact this⟩ hxb
    (fun i j ↦ by
      have h := (hattr i j).1
      have e : ∀ (k : ByteArray), ![rhsTop model k i 0, rhsTop model k i 1] j = rhsTop model k i j := by
        intro k; fin_cases j <;> rfl
      simp only [e]
      exact h)
    (fun i j ↦ (hattr i j).2) hrhsE (fun i k hk hki ↦ by subst hki; exact hhelpE i) hcompE htb ht8 hv0 hv4 hchain
  -- Key generation.
  obtain ⟨kw, kbody⟩ := h7
  have K := Stage_keygen.generatedRoot.facts ⟨_, kw⟩ kbody
  have hout6 := K.c0.hout
  subst hout6
  obtain ⟨p', hp'b, hanc, h7e, hkFb, hres⟩ := keygen_link hkind K
  have hsv := helpers_sample h6
  obtain ⟨hsC, hsI⟩ := keygen_samples K
  -- The anchors.
  obtain ⟨Eb, hEb, hW⟩ := anchor_round hs hbQ heb hr hBQ
  obtain ⟨Ec, hEc, hC⟩ := anchor_compact hs hbQ heb
  have hbits := decrypt_bits (A := keyOf (Stage_preprocess.generatedRoot.w_236_0 ⟨_, pw⟩,
      Stage_preprocess.generatedRoot.w_242_0 ⟨_, pw⟩, Stage_preprocess.generatedRoot.w_248_0 ⟨_, pw⟩,
      Stage_preprocess.generatedRoot.w_254_0 ⟨_, pw⟩, ()))
    hs (fun k ↦ by have := hmb k; omega) hem hcm hEb (by unfold roundedAnchor; exact hW) hEc (by unfold compressedAnchor; exact hC) hp'b
      (fun i ↦ (hsC i).2.2) (fun i ↦ (hsI i).2.2) (fun q ↦ (hsv q).2.2) hkFb
      (fun i ↦ by
        rw [hB0]
        have h := hres i
        rw [funext fun q ↦ (hsv q).1, (hsC i).1, (hsI i).1] at h
        exact h) hhelpE hout margin
  obtain ⟨dw, dbody⟩ := h9
  have D := Stage_decode.generatedRoot.facts ⟨_, dw⟩ dbody
  have hideal' : idl = external.input_2 := by
    obtain ⟨wi, hi⟩ := hideal
    exact hi
  rw [hideal']
  exact decode_final (hbits := hbits) D rfl rfl rfl
    (funext fun i ↦ Eq.trans (by rfl) (hanc i))
    (funext fun i ↦ (hsC i).1) rfl rfl (funext fun q ↦ (hsv q).1) rfl rfl rfl rfl h7e

/-- The average-case correctness claim of `average-w256-n32768-crt15x24-compact`: every run decrypts
to the ideal message, so the set of tapes with a wrong run is empty and has measure `0`. -/
theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have h : {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ (execution.«stage_9» = execution.«ideal»)} = ∅ := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and, not_not]
    intro exec hr
    exact run_correct hr
  rw [h, MeasureTheory.measure_empty]
  exact zero_le _

end PeProof

namespace PeCertificate

/-- The exported date correctness claim, also available as `PeProof.certificate`. -/
theorem certificate : GeneratedClaim.CorrectnessClaim := PeProof.certificate

end PeCertificate
