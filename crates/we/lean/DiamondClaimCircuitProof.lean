import DiamondProofParameters
import DiamondClaimStateProof
import DiamondCircuitInitialProof
import DiamondCircuitPublicProof

open Mxx.Primitives MxxRuntime GeneratedClaim
open DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxRecDepth 16384
set_option maxHeartbeats 3000000

def claimInitialNoise : Nat := projection * claimInjectorNoise *
  stage_0_params.diamond_preimage_max_coefficient_bound.toNat

noncomputable def claimCircuitSecret {hashModel external execution}
    (w : ClaimInjectorWitness hashModel external execution) : ExactMatrix q n 1 1 :=
  fun _ _ ↦ reducePoly q n w.commonSecret

/-- A witness ciphertext uses the actual shared terminal row and sampled preimage. -/
theorem claim_witness_slot_encoding {hashModel external execution}
    (w : ClaimInjectorWitness hashModel external execution) (slot : Fin witnessSlots)
    (key output : ExactMatrix q n 1 ell) (message : ExactMatrix q n 1 1)
    (ho : Stage_decrypt.parallel_generatedRoot_22 DiamondBackend.backend stage_1_params slot
      (w.decryptRoot.w_6_0 ⟨slot.val + 1, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val + 1 < 2; omega⟩,
        execution.stage_0.2.2.2.2.2.2.2.1 slot,
        execution.stage_0.2.2.2.2.1 ⟨slot.val + 1, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val + 1 < 2; omega⟩,
        external.input_6 ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩, ())
      (output, key, message, ())) :
    BooleanEncodingWithin (claimCircuitSecret w) key
      (if rawWitnessBits external.input_6 slot.val then 1 else 0) output claimInitialNoise := by
  obtain ⟨j, position, target, hj, hp, hrows, hpreimage, hbounded⟩ := w.witnessLinks slot
  obtain ⟨position', row, error, hp', hsecret, hstate, _, herror, hbit⟩ := w.states j
  have hpos : position' = position := by
    apply Fin.ext
    change position'.val = inputCount * stateCount + j.val at hp'
    change (position.val : Int) = (inputCount : Int) * (stateCount : Int) + j.val at hp
    have hpNat : position.val = inputCount * stateCount + j.val := by exact_mod_cast hp
    omega
  dsimp only [Stage_decrypt.parallel_generatedRoot_22,
    Stage_decrypt.parallel_generatedRoot_22.constraints_0] at ho
  obtain ⟨m, _, _, _, _, hout⟩ := ho
  have hj' : (⟨slot.val + 1, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val + 1 < 2; omega⟩ : Fin stateCount) = j := Fin.ext (by change slot.val + 1 = j.val; omega)
  have hstateEq : output = w.decryptRoot.w_6_0 j * execution.stage_0.2.2.2.2.2.2.2.1 slot := by
    simpa only [hj', matrixMul] using congrArg Prod.fst hout
  have hkeyEq : key = execution.stage_0.2.2.2.2.1 j := by
    simpa only [hj'] using congrArg (fun x ↦ x.2.1) hout
  have hrowbit : reduceMatrix q n 1 2 row 0 1 =
      claimCircuitSecret w 0 0 * (if rawWitnessBits external.input_6 slot.val then 1 else 0) := by
    have hn : j.val ≠ 0 := by omega
    have hslot : j.val - 1 = slot.val := by omega
    simpa only [hn, ite_false, hslot, claimCircuitSecret] using hbit
  apply initial_witness_encoding claimInjectorNoise
    stage_0_params.diamond_preimage_max_coefficient_bound.toNat (claimCircuitSecret w)
    (reduceMatrix q n 1 2 row) (w.decryptRoot.w_6_0 j) (w.producer.bases position) key output target
    (execution.stage_0.2.2.2.2.2.2.2.1 slot) (rawWitnessBits external.input_6 slot.val)
    (congrArg (reducePoly q n) hsecret) hrowbit
  · refine ⟨error, ?_, herror⟩
    simpa only [hpos] using hstate
  · simpa only [hkeyEq, initial_registered_gadget stage_0_params w.finalPublic] using hrows
  · exact hpreimage
  · exact hbounded
  · exact hstateEq

/-- All circuit stages use the same generated root witnesses as the injector proof. -/
theorem generated_claim_accepting_ciphertext {hashModel external execution}
    (w : ClaimInjectorWitness hashModel external execution)
    (hrun : Runs hashModel external execution) :
    ∃ (state : ExactMatrix q n 1 inner) (circuit key : ExactMatrix q n 1 ell),
      familyGetStatic w.decryptRoot.w_6_0 0 state ∧
      BooleanEncodingWithin (claimCircuitSecret w) key 1 circuit
        (factor ^ stage_1_params.depth.toNat * claimInitialNoise) ∧
      key = w.encryptRoot.w_30_0 ∧
      execution.stage_1.2.1 = state * execution.stage_0.1 -
        (state * execution.stage_0.2.2.1 +
          (state * execution.stage_0.2.2.2.1 - circuit) * execution.stage_0.2.2.2.2.2.1) := by
  obtain ⟨hvalid, henc, _, _, _, hreq, _, _, _, haccepted⟩ := hrun
  rw [haccepted] at hreq
  obtain ⟨referenceInitial, position, hreferenceInitial, hposition, haccept⟩ :=
    generated_accepting_requirement_plaintext stage_1_params requirement_2_params
      external.input_5 external.input_6 external.input_0 external.input_1 external.input_2
      external.input_3 external.input_4 rfl hreq
  have h := w.decryptRun
  dsimp only [Stage_decrypt.generatedRoot.body, Stage_decrypt.generatedRoot.constraints_0] at h
  have hstate : familyGetStatic w.decryptRoot.w_6_0 0 w.decryptRoot.w_7_0 := by tauto
  have honeGet : familyGetStatic execution.stage_0.2.2.2.2.1 0 w.decryptRoot.w_24_0 := by tauto
  have honeKey : w.decryptRoot.w_24_0 = execution.stage_0.2.2.2.2.1 0 := by
    obtain ⟨i, hi, hv⟩ := honeGet
    have hi' : i = 0 := Fin.ext (by omega)
    simpa only [hi'] using hv
  have hstate0 : w.decryptRoot.w_7_0 = w.decryptRoot.w_6_0 0 := by
    obtain ⟨i, hi, hv⟩ := hstate
    have hi' : i = 0 := Fin.ext (by omega)
    simpa only [hi'] using hv
  have hslotsRun : ∀ slot : Fin witnessSlots,
      Stage_decrypt.parallel_generatedRoot_22 DiamondBackend.backend stage_1_params slot
        (w.decryptRoot.w_6_0 ⟨slot.val + 1, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val + 1 < 2; omega⟩,
          execution.stage_0.2.2.2.2.2.2.2.1 slot,
          execution.stage_0.2.2.2.2.1 ⟨slot.val + 1, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val + 1 < 2; omega⟩,
          external.input_6 ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩, ())
        (w.decryptRoot.w_22_0 slot, w.decryptRoot.w_22_1 slot, w.decryptRoot.w_22_2 slot, ()) := by tauto
  have hinitialRun : ∀ lane : Fin circuitWidth,
      Stage_decrypt.parallel_generatedRoot_28 DiamondBackend.backend stage_1_params lane
        (stage_1_params.instance_width,
          stage_1_params.instance_width + (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count) - 1,
          0, w.decryptRoot.w_22_0, external.input_5 lane,
          w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1,
          0, w.decryptRoot.w_22_1, w.decryptRoot.w_24_0, 0, w.decryptRoot.w_22_2, 1, ())
        (w.decryptRoot.w_28_0 lane, w.decryptRoot.w_28_1 lane, w.decryptRoot.w_28_2 lane, ()) := by
    have hn := h
    simp only [matrixSub, sub_self, matrixMul] at hn
    tauto
  have hloop : MxxIR.IterRuns
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_33 DiamondBackend.backend
        stage_1_params layer (current.1, current.2.1, current.2.2.1,
          external.input_0, w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1,
          external.input_1, external.input_2, external.input_3, w.decryptRoot.w_24_0, 1, ()) next)
      stage_1_params.depth.toNat
      (w.decryptRoot.w_28_0, w.decryptRoot.w_28_1, w.decryptRoot.w_28_2, ())
      (w.decryptRoot.w_33_0, w.decryptRoot.w_33_1, w.decryptRoot.w_33_2, ()) := by tauto
  have hsource : familyGetStatic external.input_4 0 w.decryptRoot.w_35_0 := by tauto
  have hcircuit : familyGetDynamic w.decryptRoot.w_33_0 w.decryptRoot.w_35_0 w.decryptRoot.w_36_0 := by tauto
  have hresidual : execution.stage_1.2.1 = w.decryptRoot.w_7_0 * execution.stage_0.1 -
      (w.decryptRoot.w_7_0 * execution.stage_0.2.2.1 +
        (w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1 - w.decryptRoot.w_36_0) *
          execution.stage_0.2.2.2.2.2.1) := by
    repeat' obtain ⟨_, h⟩ := h
    rw [h]
    rfl
  have hmessages : ∀ slot : Fin witnessSlots, w.decryptRoot.w_22_2 slot 0 0 =
      if external.input_6 ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩ = 1
        then 1 else 0 := by
    intro slot
    have hs := hslotsRun slot
    dsimp only [Stage_decrypt.parallel_generatedRoot_22,
      Stage_decrypt.parallel_generatedRoot_22.constraints_0] at hs
    obtain ⟨message, _, _, _, hselect, hout⟩ := hs
    have hm := congrArg (fun x ↦ x.2.2.1) hout
    dsimp only at hm
    obtain ⟨position, hp, hv⟩ := hselect
    change Fin 2 at position
    rw [hm, hv]
    fin_cases position
    · have hz : external.input_6 ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩ = 0 := hp.symm
      rw [hz]
      simp
    · have hz : external.input_6 ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩ = 1 := hp.symm
      rw [hz]
      simp
  have hplain : CircuitPlaintextAgrees
      (w.decryptRoot.w_28_0, w.decryptRoot.w_28_1, w.decryptRoot.w_28_2, ()) referenceInitial := by
    intro lane
    obtain ⟨facts⟩ := generated_initial_lane_facts DiamondBackend.backend stage_1_params lane
      stage_1_params.instance_width
      (stage_1_params.instance_width + (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count) - 1)
      (external.input_5 lane) 0 (w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1)
      0 w.decryptRoot.w_24_0 0 1 w.decryptRoot.w_22_0 w.decryptRoot.w_22_1
      w.decryptRoot.w_22_2 _ (hinitialRun lane)
    exact generated_initial_plaintext_lane_agrees requirement_2_params lane
      stage_1_params.instance_width
      (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count)
      (external.input_5 lane) _ _ _ _ _ _ _ external.input_6 _ (referenceInitial lane)
      facts (by have hh := hvalid.2.2.2.2.2.1 lane; omega) hmessages (hreferenceInitial lane)
  have hmessage := haccept
    (w.decryptRoot.w_28_0, w.decryptRoot.w_28_1, w.decryptRoot.w_28_2, ())
    (w.decryptRoot.w_33_0, w.decryptRoot.w_33_1, w.decryptRoot.w_33_2, ())
    (w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1) w.decryptRoot.w_24_0 1
    (by simp) hplain hloop
  change w.decryptRoot.w_33_2 position 0 0 = 1 at hmessage
  obtain ⟨selector, hsecret, _, hzero⟩ := claim_zero_state_encoding w
  have hone : BooleanEncodingWithin (claimCircuitSecret w) w.decryptRoot.w_24_0 1
      (w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1) claimInitialNoise := by
    rw [honeKey, hstate0]
    exact initial_one_encoding stage_0_params w.finalPublic (claimCircuitSecret w)
      selector _ claimInjectorNoise stage_0_params.diamond_preimage_max_coefficient_bound.toNat
      hsecret hzero (generated_final_preimages_bounded DiamondBackend.backend hashModel
        stage_0_params henc).2.2
  have hslots : ∀ slot : Fin witnessSlots,
      BooleanEncodingWithin (claimCircuitSecret w) (w.decryptRoot.w_22_1 slot)
        (w.decryptRoot.w_22_2 slot 0 0) (w.decryptRoot.w_22_0 slot) claimInitialNoise ∧
      (w.decryptRoot.w_22_2 slot 0 0 = 0 ∨ w.decryptRoot.w_22_2 slot 0 0 = 1) := by
    intro slot
    have hs := claim_witness_slot_encoding w slot _ _ _ (hslotsRun slot)
    have hbit : rawWitnessBits external.input_6 slot.val =
        decide (external.input_6 ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩ = 1) := by
      exact rawWitnessBits_at external.input_6
        ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩
    rw [hmessages slot]
    constructor
    · simpa only [hbit, decide_eq_true_eq] using hs
    · split <;> simp
  have hinitial : CircuitStateWithin (claimCircuitSecret w) claimInitialNoise
      (w.decryptRoot.w_28_0, w.decryptRoot.w_28_1, w.decryptRoot.w_28_2, ()) := by
    intro lane
    exact generated_initial_lane_within DiamondBackend.backend stage_1_params lane claimInitialNoise
      _ _ _ (claimCircuitSecret w) _ _ _ _ _ _ hone hslots (hinitialRun lane)
  have hfinal := generated_circuit_iteration_within stage_1_params stage_1_params.depth.toNat
    claimInitialNoise (claimCircuitSecret w)
    (w.decryptRoot.w_28_0, w.decryptRoot.w_28_1, w.decryptRoot.w_28_2, ())
    (w.decryptRoot.w_33_0, w.decryptRoot.w_33_1, w.decryptRoot.w_33_2, ())
    external.input_0 external.input_1 external.input_2 external.input_3
    (w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1) w.decryptRoot.w_24_0 1
    hone (by simp) hinitial hloop
  have eh := w.encryptRun
  dsimp only [Stage_encrypt.generatedRoot.body, Stage_encrypt.generatedRoot.constraints_0,
    Stage_encrypt.generatedRoot.constraints_1] at eh
  have hepublic : execution.stage_0.2.2.2.2.1 = w.encryptRoot.w_8_0 := by
    have hh := eh
    repeat' obtain ⟨_, hh⟩ := hh
    rw [hh]
  have heoneGet : familyGetStatic w.encryptRoot.w_8_0 0 w.encryptRoot.w_9_0 := by tauto
  have heone : w.encryptRoot.w_9_0 = w.decryptRoot.w_24_0 := by
    obtain ⟨i, hi, hv⟩ := heoneGet
    have hi' : i = 0 := Fin.ext (by omega)
    simpa only [hi', ← hepublic, ← honeKey] using hv
  have heindex : ∀ lane : Fin circuitWidth,
      Stage_encrypt.parallel_generatedRoot_20 DiamondBackend.backend hashModel stage_0_params lane
        (stage_1_params.instance_width,
          stage_1_params.instance_width + (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count) - 1, ())
        (w.encryptRoot.w_20_0 lane) := by
    simpa only [add_zero] using
      (show ∀ lane : Fin circuitWidth, Stage_encrypt.parallel_generatedRoot_20 _ _ _ _ _ _ from by tauto)
  have heinitialRun : ∀ lane : Fin circuitWidth,
      Stage_encrypt.parallel_generatedRoot_22 DiamondBackend.backend hashModel stage_0_params lane
        (stage_1_params.instance_width,
          stage_1_params.instance_width + (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count) - 1,
          0, w.encryptRoot.w_8_0, w.encryptRoot.w_20_0 lane, external.input_5 lane,
          w.encryptRoot.w_9_0, ()) (w.encryptRoot.w_22_0 lane) := by
    have hn := eh
    simp only [matrixSub, sub_self] at hn
    tauto
  have heloop : MxxIR.IterRuns
      (fun layer current next ↦ Stage_encrypt.sequential_generatedRoot_27 DiamondBackend.backend
        hashModel stage_0_params layer (current, external.input_0, w.encryptRoot.w_9_0,
          external.input_1, external.input_2, external.input_3, ()) next)
      stage_0_params.depth.toNat w.encryptRoot.w_22_0 w.encryptRoot.w_27_0 := by tauto
  have hesource : familyGetStatic external.input_4 0 w.encryptRoot.w_29_0 := by tauto
  have hecircuit : familyGetDynamic w.encryptRoot.w_27_0 w.encryptRoot.w_29_0 w.encryptRoot.w_30_0 := by tauto
  have heinitial : w.decryptRoot.w_28_1 = w.encryptRoot.w_22_0 := by
    funext lane
    obtain ⟨facts⟩ := generated_initial_lane_facts DiamondBackend.backend stage_1_params lane
      _ _ _ _ _ _ _ _ _ _ _ _ _ (hinitialRun lane)
    have hkeys : ∀ slot : Fin witnessSlots, w.decryptRoot.w_22_1 slot =
        w.encryptRoot.w_8_0 ⟨slot.val + 1, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val + 1 < 2; omega⟩ := by
      intro slot
      have hs := hslotsRun slot
      dsimp only [Stage_decrypt.parallel_generatedRoot_22,
        Stage_decrypt.parallel_generatedRoot_22.constraints_0] at hs
      obtain ⟨_, _, _, _, _, hout⟩ := hs
      simpa only [hepublic] using congrArg (fun x ↦ x.2.1) hout
    have honePublic : w.decryptRoot.w_24_0 = w.encryptRoot.w_8_0 0 := honeKey.trans (congrFun hepublic 0)
    apply generated_initial_public_lane_agrees DiamondBackend.backend hashModel stage_0_params
      lane _ _ (external.input_5 lane) _ 0 (w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1)
      w.encryptRoot.w_8_0 w.decryptRoot.w_22_0 w.decryptRoot.w_22_1 w.decryptRoot.w_22_2
      (w.decryptRoot.w_28_0 lane, w.decryptRoot.w_28_1 lane, w.decryptRoot.w_28_2 lane, ())
      (w.encryptRoot.w_22_0 lane)
      (by simpa only [honePublic] using facts) hkeys (heindex lane)
    simpa only [heone, honePublic] using heinitialRun lane
  have hpublicFinal : w.decryptRoot.w_33_1 = w.encryptRoot.w_27_0 :=
    generated_circuit_public_iteration_agrees DiamondBackend.backend hashModel stage_1_params
      stage_0_params stage_1_params.depth.toNat
      (w.decryptRoot.w_28_0, w.decryptRoot.w_28_1, w.decryptRoot.w_28_2, ())
      (w.decryptRoot.w_33_0, w.decryptRoot.w_33_1, w.decryptRoot.w_33_2, ())
      w.encryptRoot.w_22_0 w.encryptRoot.w_27_0
      external.input_0 external.input_1 external.input_2 external.input_3
      (w.decryptRoot.w_7_0 * execution.stage_0.2.2.2.1) w.decryptRoot.w_24_0 1
      rfl rfl rfl heinitial hloop (by simpa only [heone] using heloop)
  obtain ⟨ep, _, hep⟩ := hesource
  have hep0 : ep = 0 := Subsingleton.elim _ _
  have heposition : w.encryptRoot.w_29_0 = external.input_4 0 := by simpa only [hep0] using hep
  have hkeyFinal : w.decryptRoot.w_33_1 position = w.encryptRoot.w_30_0 := by
    rw [hpublicFinal]
    exact circuit_lookup_unique ⟨position, hposition.trans heposition.symm, rfl⟩ hecircuit
  obtain ⟨sp, _, hsp⟩ := hsource
  have hsp0 : sp = 0 := Subsingleton.elim _ _
  have hsourceIndex : w.decryptRoot.w_35_0 = external.input_4 0 := by simpa only [hsp0] using hsp
  have hc : w.decryptRoot.w_36_0 = w.decryptRoot.w_33_0 position := circuit_lookup_unique hcircuit
    ⟨position, hposition.trans hsourceIndex.symm, rfl⟩
  refine ⟨w.decryptRoot.w_7_0, w.decryptRoot.w_36_0, w.decryptRoot.w_33_1 position,
    hstate, ?_, hkeyFinal, hresidual⟩
  have hf := (hfinal position).1
  simpa only [hc, hmessage] using hf

/-- The linked runs yield one correlated internal witness and the observed residual. -/
theorem generated_claim_circuit (hashModel : HashModel) (external : ExternalInputs)
    (execution : Execution) (hrun : Runs hashModel external execution) :
    ∃ w : ClaimInjectorWitness hashModel external execution,
      ∃ (state : ExactMatrix q n 1 inner) (circuit : ExactMatrix q n 1 ell),
        familyGetStatic w.decryptRoot.w_6_0 0 state ∧
        BooleanEncodingWithin (claimCircuitSecret w) w.encryptRoot.w_30_0 1 circuit
          (factor ^ stage_1_params.depth.toNat * claimInitialNoise) ∧
        execution.stage_1.2.1 = state * execution.stage_0.1 -
          (state * execution.stage_0.2.2.1 +
            (state * execution.stage_0.2.2.2.1 - circuit) * execution.stage_0.2.2.2.2.2.1) := by
  obtain ⟨w⟩ := generated_claim_injector hashModel external execution hrun
  obtain ⟨state, circuit, key, hstate, hencoding, hkey, hresidual⟩ :=
    generated_claim_accepting_ciphertext w hrun
  exact ⟨w, state, circuit, hstate, hkey ▸ hencoding, hresidual⟩

#print axioms claim_witness_slot_encoding
#print axioms generated_claim_accepting_ciphertext
#print axioms generated_claim_circuit

end DiamondGeneratedProof
