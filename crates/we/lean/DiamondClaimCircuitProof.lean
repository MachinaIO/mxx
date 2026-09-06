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

theorem claim_witness_slot_encoding {hashModel external execution}
    (w : ClaimInjectorWitness hashModel external execution) (slot : Fin witnessSlots)
    (stateIndex keyIndex : Int) (state : ExactMatrix q n 1 inner)
    (key output : ExactMatrix q n 1 ell)
    (hi : Stage_decrypt.parallel_generatedRoot_28 DiamondBackend.backend stage_1_params slot
      () stateIndex)
    (hs : Stage_decrypt.parallel_generatedRoot_29 DiamondBackend.backend stage_1_params slot
      (stateIndex, execution.stage_1.2.2.2.2.2.2.1, ()) state)
    (hki : Stage_decrypt.parallel_generatedRoot_43 DiamondBackend.backend stage_1_params slot
      () keyIndex)
    (hk : Stage_decrypt.parallel_generatedRoot_44 DiamondBackend.backend stage_1_params slot
      (keyIndex, execution.stage_0.2.2.2.2.1, ()) key)
    (ho : Stage_decrypt.parallel_generatedRoot_31 DiamondBackend.backend stage_1_params slot
      (state, execution.stage_0.2.2.2.2.2.2.2.1 slot, ()) output) :
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
  obtain ⟨sv, _, _, ⟨si, hsi, hsv⟩, hso⟩ := hs
  obtain ⟨kv, _, _, ⟨ki, hki', hkv⟩, hko⟩ := hk
  have hsi' : si = j := Fin.ext (by change stateIndex = (slot.val : Int) + 1 at hi; omega)
  have hki'' : ki = j := Fin.ext (by change keyIndex = (slot.val : Int) + 1 at hki; omega)
  have hstateEq : state = execution.stage_1.2.2.2.2.2.2.1 j := by
    simpa only [hsi'] using hso.trans hsv
  have hkeyEq : key = execution.stage_0.2.2.2.2.1 j := by
    simpa only [hki''] using hko.trans hkv
  have hrowbit : reduceMatrix q n 1 2 row 0 1 =
      claimCircuitSecret w 0 0 * (if rawWitnessBits external.input_6 slot.val then 1 else 0) := by
    have hn : j.val ≠ 0 := by omega
    have hj' : j.val - 1 = slot.val := by omega
    simpa only [hn, ite_false, hj', claimCircuitSecret] using hbit
  apply initial_witness_encoding stage_1_params slot claimInjectorNoise
    stage_0_params.diamond_preimage_max_coefficient_bound.toNat (claimCircuitSecret w)
    (reduceMatrix q n 1 2 row) state (w.producer.bases position) key output target
    (execution.stage_0.2.2.2.2.2.2.2.1 slot) (rawWitnessBits external.input_6 slot.val)
    (congrArg (reducePoly q n) hsecret) hrowbit
  · refine ⟨error, ?_, herror⟩
    simpa only [hstateEq, hpos] using hstate
  · simpa only [hkeyEq, initial_registered_gadget stage_0_params w.finalPublic] using hrows
  · exact hpreimage
  · exact hbounded
  · exact ho

theorem claim_initial_public_lane (hashModel : HashModel) (lane : Nat)
    (start last instanceBit instanceFlag witnessFlag dindex eindex : Int)
    (keys : Fin stateCount → ExactMatrix q n 1 ell)
    (one dkey ekey dwitness dinstance dout ewitness eout : ExactMatrix q n 1 ell)
    (hif : Stage_decrypt.parallel_generatedRoot_19 DiamondBackend.backend stage_1_params
      lane start instanceFlag)
    (hwf : Stage_decrypt.parallel_generatedRoot_24 DiamondBackend.backend stage_1_params
      lane (start, last, ()) witnessFlag)
    (hdi : Stage_decrypt.parallel_generatedRoot_27 DiamondBackend.backend stage_1_params
      lane (start, last, ()) dindex)
    (hei : Stage_encrypt.parallel_generatedRoot_19 DiamondBackend.backend hashModel stage_0_params
      lane (start, last, ()) eindex)
    (hdk : familyGetDynamic keys (dindex + 1) dkey)
    (hek : Stage_encrypt.parallel_generatedRoot_20 DiamondBackend.backend hashModel stage_0_params
      lane (eindex, keys, ()) ekey)
    (hdw : Stage_decrypt.parallel_generatedRoot_46 DiamondBackend.backend stage_1_params
      lane (witnessFlag, 0, dkey, ()) dwitness)
    (hds : Stage_decrypt.parallel_generatedRoot_49 DiamondBackend.backend stage_1_params
      lane (instanceBit, 0, one, ()) dinstance)
    (hdo : Stage_decrypt.parallel_generatedRoot_50 DiamondBackend.backend stage_1_params
      lane (instanceFlag, dwitness, dinstance, ()) dout)
    (hew : Stage_encrypt.parallel_generatedRoot_26 DiamondBackend.backend hashModel stage_0_params
      lane (ekey, start, 0, last, ()) ewitness)
    (heo : Stage_encrypt.parallel_generatedRoot_27 DiamondBackend.backend hashModel stage_0_params
      lane (instanceBit, ewitness, start, 0, one, ()) eout) : dout = eout := by
  dsimp only [Stage_decrypt.parallel_generatedRoot_19] at hif
  dsimp only [Stage_decrypt.parallel_generatedRoot_24] at hwf
  obtain ⟨di, _, _, _, hdi, hdio⟩ := hdi
  obtain ⟨ei, _, _, _, hei, heio⟩ := hei
  obtain ⟨ek, _, _, heget, heko⟩ := hek
  obtain ⟨dw, _, _, _, hdw, hdwo⟩ := hdw
  obtain ⟨ds, _, _, _, hds, hdso⟩ := hds
  obtain ⟨do', _, _, _, hdo, hdoo⟩ := hdo
  obtain ⟨ew, ew', _, _, _, hew, _, _, _, hew', hewo⟩ := hew
  obtain ⟨es, eo, _, _, _, hes, _, _, _, heo, heoo⟩ := heo
  have hdiv := initial_select_value _ _ _ _ hdi
  have heiv := initial_select_value _ _ _ _ hei
  have hdwv := initial_select_value _ _ _ _ hdw
  have hdsv := initial_select_value _ _ _ _ hds
  have hdov := initial_select_value _ _ _ _ hdo
  have hewv := initial_select_value _ _ _ _ hew
  have hewv' := initial_select_value _ _ _ _ hew'
  have hesv := initial_select_value _ _ _ _ hes
  have heov := initial_select_value _ _ _ _ heo
  by_cases hstart : start ≤ (lane : Int)
  · have hi : ¬ (lane : Int) ≤ start - 1 := by omega
    by_cases hend : (lane : Int) ≤ last
    · have haddr : eindex = dindex + 1 := by simp_all
      have hkey : ek = dkey := circuit_lookup_unique (haddr ▸ heget) hdk
      simp_all
    · simp_all
  · have hi : (lane : Int) ≤ start - 1 := by omega
    simp_all

/-- The actual selected ciphertext has accepting plaintext and a propagated integer
    error witness, using only the linked runs and their jointly derived injector data. -/
theorem generated_claim_accepting_ciphertext {hashModel external execution}
    (w : ClaimInjectorWitness hashModel external execution)
    (hrun : Runs hashModel external execution) :
    ∃ (state : ExactMatrix q n 1 inner) (circuit key : ExactMatrix q n 1 ell),
      familyGetStatic execution.stage_1.2.2.2.2.2.2.1 0 state ∧
      BooleanEncodingWithin (claimCircuitSecret w) key 1 circuit
        (factor ^ stage_1_params.depth.toNat * claimInitialNoise) ∧
      key = execution.stage_0.2.2.2.2.2.2.2.2.2.1 ∧
      execution.stage_1.2.2.2.2.1 = state * execution.stage_0.1 -
        (state * execution.stage_0.2.2.1 +
          (state * execution.stage_0.2.2.2.1 - circuit) * execution.stage_0.2.2.2.2.2.1) := by
  obtain ⟨hvalid, henc, hdec, _, _, hreq, _, _, _, haccepted⟩ := hrun
  rw [haccepted] at hreq
  obtain ⟨referenceIndices, referenceWitnesses, referenceInitial, position,
    hreferenceIndices, hreferenceWitnesses, hreferenceInitial, hposition, haccept⟩ :=
    generated_accepting_requirement_plaintext stage_1_params requirement_2_params
      external.input_5 external.input_6 external.input_0 external.input_1 external.input_2
      external.input_3 external.input_4 rfl hreq
  dsimp only [Stage_decrypt.generatedRoot] at hdec
  rcases hdec with ⟨w2, w3, w5, w6, w8, state, w19, w24, w26, w27, w28, w29,
    w31, w32, w33, w35, w36, w37, w38, w40, w42, w43, w44, w45, w46, w47,
    w48, w49, w50, w53, w54, w55, w56, w57, w58, w59, w60, w61, w62,
    w67a, w67b, w67c, w70, circuit, coefficient, h⟩
  rcases h with ⟨_, _, _, _, h3, _, h5, _, _, _, _, hstate,
    _, h19, _, h24, _, h26, _, h27, _, h28, _, h29, _, h31, _, h32,
    _, h33, _, h35, _, h36, _, h37, _, h38, h40, _, h42, _, h43, _, h44,
    _, h45, _, h46, _, h47, _, h48, _, h49, _, h50, _, h53, _, h54,
    _, h55, _, h56, _, h57, _, h58, _, h59, _, h60, _, h61, _, h62,
    _, hloop, _, _, hsource, _, _, hcircuit, _, hout⟩
  dsimp only [Stage_decrypt.parallel_generatedRoot_26] at h26
  dsimp only [Stage_decrypt.parallel_generatedRoot_35] at h35
  dsimp only [Stage_decrypt.parallel_generatedRoot_36] at h36
  dsimp only [Stage_decrypt.parallel_generatedRoot_42] at h42
  dsimp only [Stage_decrypt.parallel_generatedRoot_47] at h47
  dsimp only [Stage_decrypt.parallel_generatedRoot_48] at h48
  dsimp only [Stage_decrypt.parallel_generatedRoot_53] at h53
  dsimp only [Stage_decrypt.parallel_generatedRoot_54] at h54
  dsimp only [Stage_decrypt.parallel_generatedRoot_55] at h55
  dsimp only [Stage_decrypt.parallel_generatedRoot_59] at h59
  dsimp only [Stage_decrypt.parallel_generatedRoot_60] at h60
  have hw8 : execution.stage_1.2.2.2.2.2.2.1 = w8 := by rw [hout]
  have hstate0 : state = execution.stage_1.2.2.2.2.2.2.1 0 := by
    obtain ⟨i, hi, hv⟩ := hstate
    have hi' : i = 0 := Fin.ext (by omega)
    simpa only [hi', hw8] using hv
  have honeKey : w40 = execution.stage_0.2.2.2.2.1 0 := by
    obtain ⟨i, hi, hv⟩ := h40
    have hi' : i = 0 := Fin.ext (by omega)
    simpa only [hi'] using hv
  have hbits : ∀ slot : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_56 DiamondBackend.backend
      stage_1_params slot (w5 slot, 0, 1, ()) (w56 slot) := by
    intro slot
    simpa only [h54 slot, h55 slot] using h56 slot
  have hraw : ∀ lane, external.input_5 lane = 0 ∨ external.input_5 lane = 1 := by
    intro lane
    have hh := hvalid.2.2.2.2.2.1 lane
    omega
  have hplain : CircuitPlaintextAgrees (w38, w50, w62, ()) referenceInitial := by
    apply generated_initial_plaintext_agrees DiamondBackend.backend stage_1_params
      requirement_2_params stage_1_params.instance_width
      (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count)
      (w38, w50, w62, ()) external.input_5 external.input_6 w19 w24 w27 referenceIndices
      referenceWitnesses w3 w5 w56 w57 w58 w61 referenceInitial (by decide) hraw
      h3 h5 hbits
    · simpa only [add_zero] using h27
    · exact hreferenceIndices
    · exact h57
    · exact hreferenceWitnesses
    · simpa only [add_zero] using h19
    · simpa only [add_zero] using h24
    · intro lane
      simpa only [h53 lane, matrixSub, sub_self] using h58 lane
    · intro lane
      simpa only [h59 lane, h60 lane, matrixSub, sub_self] using h61 lane
    · exact h62
    · exact hreferenceInitial
  have hmessage := haccept (w38, w50, w62, ()) (w67a, w67b, w67c, ())
    (state * execution.stage_0.2.2.2.1) w40 1 (by simp) hplain hloop
  change w67c position 0 0 = 1 at hmessage
  obtain ⟨selector, hsecret, _, hzero⟩ := claim_zero_state_encoding w
  have hone : BooleanEncodingWithin (claimCircuitSecret w) w40 1
      (state * execution.stage_0.2.2.2.1) claimInitialNoise := by
    rw [honeKey, hstate0]
    exact initial_one_encoding stage_0_params w.finalPublic (claimCircuitSecret w)
      selector _ claimInjectorNoise stage_0_params.diamond_preimage_max_coefficient_bound.toNat
      hsecret hzero (generated_final_preimages_bounded DiamondBackend.backend hashModel
        stage_0_params henc).2.2
  have hslots : ∀ slot : Fin witnessSlots, BooleanEncodingWithin (claimCircuitSecret w) (w44 slot)
      (w56 slot 0 0) (w31 slot) claimInitialNoise ∧
      (w56 slot 0 0 = 0 ∨ w56 slot 0 0 = 1) := by
    intro slot
    have hs := claim_witness_slot_encoding w slot (w28 slot) (w43 slot) (w29 slot)
      (w44 slot) (w31 slot) (h28 slot) (by simpa only [hw8] using h29 slot)
      (h43 slot) (h44 slot) (h31 slot)
    obtain ⟨rawPosition, hp, hv⟩ := generated_witness_prefix DiamondBackend.backend
      stage_1_params slot external.input_6 (w3 slot) (w5 slot) (h3 slot) (h5 slot)
    have hb : rawWitnessBits external.input_6 slot.val = decide (w5 slot = 1) := by
      rw [← hp, rawWitnessBits_at, hv]
    obtain ⟨bitMessage, _, _, _, ⟨bitPosition, hbitPosition, hbitValue⟩, hbout⟩ := hbits slot
    change Fin 2 at bitPosition
    have hm : w56 slot 0 0 = if rawWitnessBits external.input_6 slot.val then 1 else 0 := by
      rw [hbout, hbitValue, hb]
      fin_cases bitPosition
      · have hz : w5 slot = 0 := hbitPosition.symm
        simp [hz]
      · have ho : w5 slot = 1 := hbitPosition.symm
        simp [ho]
    refine ⟨hm ▸ hs, ?_⟩
    rw [hm]
    split <;> simp
  have hinitial : CircuitStateWithin (claimCircuitSecret w) claimInitialNoise
      (w38, w50, w62, ()) := by
    intro lane
    obtain ⟨hw, hwb⟩ := initial_gathered_triple_within DiamondBackend.backend stage_1_params
      lane claimInitialNoise (w27 lane) (claimCircuitSecret w) w31 w44 w56
      (w32 lane) (w45 lane) (w57 lane) hslots (h32 lane) (h45 lane) (h57 lane)
    apply generated_initial_lane_within DiamondBackend.backend stage_1_params lane
      claimInitialNoise (w19 lane) (w24 lane) (external.input_5 lane) (claimCircuitSecret w)
      (state * execution.stage_0.2.2.2.1) w40 (w32 lane) (w45 lane) (w33 lane)
      (w46 lane) (w37 lane) (w49 lane) (w38 lane) (w50 lane) (w57 lane)
      (w58 lane) (w61 lane) (w62 lane) hone hw hwb
    · simpa only [h26 lane] using h33 lane
    · simpa only [h42 lane] using h46 lane
    · simpa only [h53 lane, matrixSub, sub_self] using h58 lane
    · simpa only [h35 lane, h36 lane] using h37 lane
    · simpa only [h47 lane, h48 lane] using h49 lane
    · simpa only [h59 lane, h60 lane, matrixSub, sub_self] using h61 lane
    · exact h38 lane
    · exact h50 lane
    · exact h62 lane
  have hfinal := generated_circuit_iteration_within stage_1_params stage_1_params.depth.toNat
    claimInitialNoise (claimCircuitSecret w) (w38, w50, w62, ()) (w67a, w67b, w67c, ())
    external.input_0 external.input_1 external.input_2 external.input_3
    (state * execution.stage_0.2.2.2.1) w40 1 hone (by simp) hinitial hloop
  dsimp only [Stage_encrypt.generatedRoot] at henc
  rcases henc with ⟨e0, e1, et1, e2, et2, ebase, etrapdoor, e6, euniform, e8,
    epublicInputs, eone, e19, e20, e26, e27, e32, e35, eCircuit, e38, etarget,
    edigits, edecoderTarget, edecoder, e46, e51, e52, e53, e55, ekeyTarget,
    ekey, egadget, eoneTarget, eonePreimage, e65, e66, et66, e67, e68, e69,
    e70, e71, e72, e73, e74, e75, et75, e76, e77, e78, eh⟩
  have heoutputs : execution.stage_0.2.2.2.2.1 = epublicInputs ∧
      execution.stage_0.2.2.2.2.2.2.2.2.2.1 = eCircuit := by
    have hh := eh
    repeat' obtain ⟨_, hh⟩ := hh
    constructor <;> rw [hh]
  have heoneGet : familyGetStatic epublicInputs 0 eone := by tauto
  have heone : eone = w40 := by
    obtain ⟨i, hi, hv⟩ := heoneGet
    have hi' : i = 0 := Fin.ext (by omega)
    simpa only [hi', ← heoutputs.1, ← honeKey] using hv
  have he19 : ∀ lane : Fin circuitWidth, Stage_encrypt.parallel_generatedRoot_19 DiamondBackend.backend
      hashModel stage_0_params lane (stage_1_params.instance_width + 0,
        stage_1_params.instance_width + 0 +
          (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count - 1), ())
        (e19 lane) := by tauto
  have he20 : ∀ lane : Fin circuitWidth, Stage_encrypt.parallel_generatedRoot_20 DiamondBackend.backend
      hashModel stage_0_params lane (e19 lane, epublicInputs, ()) (e20 lane) := by tauto
  have he26 : ∀ lane : Fin circuitWidth, Stage_encrypt.parallel_generatedRoot_26 DiamondBackend.backend
      hashModel stage_0_params lane (e20 lane, stage_1_params.instance_width + 0,
        eone - eone, stage_1_params.instance_width + 0 +
          (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count - 1), ())
        (e26 lane) := by tauto
  have he27 : ∀ lane : Fin circuitWidth, Stage_encrypt.parallel_generatedRoot_27 DiamondBackend.backend
      hashModel stage_0_params lane (external.input_5 lane, e26 lane,
        stage_1_params.instance_width + 0, eone - eone, eone, ()) (e27 lane) := by tauto
  have heloop : MxxIR.IterRuns
      (fun layer current next ↦ Stage_encrypt.sequential_generatedRoot_32 DiamondBackend.backend
        hashModel stage_0_params layer (current, external.input_0, external.input_1,
          external.input_2, external.input_3, eone, ()) next)
        stage_0_params.depth.toNat e27 e32 := by tauto
  have hesource : familyGetDynamic external.input_4 0 e35 := by tauto
  have hecircuit : familyGetDynamic e32 e35 eCircuit := by tauto
  have heinitial : w50 = e27 := by
    funext lane
    obtain ⟨dk, _, _, ⟨slot, hslot, hslotValue⟩, hdk⟩ := h45 lane
    obtain ⟨pk, _, _, hpk, hpkout⟩ := h44 slot
    have hshift : w43 slot = w27 lane + 1 := by
      have hh := h43 slot
      change w43 slot = (slot.val : Int) + 1 at hh
      simpa only [hslot] using hh
    have hdkget : familyGetDynamic epublicInputs (w27 lane + 1) (w45 lane) := by
      simpa only [heoutputs.1, hshift, hdk, hslotValue, hpkout] using hpk
    apply claim_initial_public_lane hashModel lane (stage_1_params.instance_width + 0)
      (stage_1_params.instance_width + 0 +
        (stage_1_params.diamond_batch_bits * stage_1_params.diamond_input_count - 1))
      (external.input_5 lane) (w19 lane) (w24 lane) (w27 lane) (e19 lane) epublicInputs
      w40 (w45 lane) (e20 lane) (w46 lane) (w49 lane) (w50 lane) (e26 lane) (e27 lane)
      (h19 lane) (h24 lane) (h27 lane) (he19 lane) hdkget (he20 lane)
    · simpa only [h42 lane, matrixSub, sub_self] using h46 lane
    · simpa only [h47 lane, h48 lane, matrixSub, sub_self] using h49 lane
    · exact h50 lane
    · simpa only [sub_self] using he26 lane
    · simpa only [sub_self, heone] using he27 lane
  have hpublicFinal : w67b = e32 := generated_circuit_public_iteration_agrees
    DiamondBackend.backend hashModel stage_1_params stage_0_params stage_1_params.depth.toNat
    (w38, w50, w62, ()) (w67a, w67b, w67c, ()) e27 e32
    external.input_0 external.input_1 external.input_2 external.input_3
    (state * execution.stage_0.2.2.2.1) w40 1 rfl rfl rfl heinitial hloop
    (by simpa only [heone] using heloop)
  obtain ⟨ep, _, hep⟩ := hesource
  have hep0 : ep = 0 := Subsingleton.elim _ _
  have heindex : e35 = external.input_4 0 := by simpa only [hep0] using hep
  have hkeyFinal : w67b position = execution.stage_0.2.2.2.2.2.2.2.2.2.1 := by
    rw [heoutputs.2, hpublicFinal]
    exact circuit_lookup_unique ⟨position, hposition.trans heindex.symm, rfl⟩ hecircuit
  obtain ⟨sourcePosition, _, hsourceValue⟩ := hsource
  have hsp : sourcePosition = 0 := Subsingleton.elim _ _
  have hsourceIndex : w70 = external.input_4 0 := by simpa only [hsp] using hsourceValue
  have hc : circuit = w67a position := circuit_lookup_unique hcircuit
    ⟨position, hposition.trans hsourceIndex.symm, rfl⟩
  refine ⟨state, circuit, w67b position, ?_, ?_, hkeyFinal, ?_⟩
  · simpa only [hw8] using hstate
  · have hf := (hfinal position).1
    change BooleanEncodingWithin (claimCircuitSecret w) (w67b position)
      (w67c position 0 0) (w67a position)
      (factor ^ stage_1_params.depth.toNat * claimInitialNoise) at hf
    simpa only [hc, hmessage] using hf
  · rw [hout]
    rfl

/-- From the generated linked Runs alone: one jointly derived injector witness, the
    actual selected ciphertext, the exact encryption artifact public key, and the actual
    residual expression. Numeric decoder clearance is deliberately not assumed here. -/
theorem generated_claim_circuit (hashModel : HashModel) (external : ExternalInputs)
    (execution : Execution) (hrun : Runs hashModel external execution) :
    ∃ w : ClaimInjectorWitness hashModel external execution,
      ∃ (state : ExactMatrix q n 1 inner) (circuit : ExactMatrix q n 1 ell),
        familyGetStatic execution.stage_1.2.2.2.2.2.2.1 0 state ∧
        BooleanEncodingWithin (claimCircuitSecret w)
          execution.stage_0.2.2.2.2.2.2.2.2.2.1 1 circuit
          (factor ^ stage_1_params.depth.toNat * claimInitialNoise) ∧
        execution.stage_1.2.2.2.2.1 = state * execution.stage_0.1 -
          (state * execution.stage_0.2.2.1 +
            (state * execution.stage_0.2.2.2.1 - circuit) * execution.stage_0.2.2.2.2.2.1) := by
  obtain ⟨w⟩ := generated_claim_injector hashModel external execution hrun
  obtain ⟨state, circuit, key, hstate, hencoding, hkey, hresidual⟩ :=
    generated_claim_accepting_ciphertext w hrun
  exact ⟨w, state, circuit, hstate, hkey ▸ hencoding, hresidual⟩

#print axioms claim_witness_slot_encoding
#print axioms claim_initial_public_lane
#print axioms generated_claim_accepting_ciphertext
#print axioms generated_claim_circuit

end DiamondGeneratedProof
