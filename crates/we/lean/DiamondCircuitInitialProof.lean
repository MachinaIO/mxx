import DiamondProofParameters
import DiamondCircuitRequirementProof
import DiamondFinalEncodingProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxHeartbeats 2000000
set_option maxRecDepth 8192

theorem initial_registered_gadget (params : Stage_encrypt.Params)
    {decoder key one digits half publicInputs publicCircuit}
    (w : FinalPublicWitness DiamondBackend.backend params decoder key one digits half
      publicInputs publicCircuit) : w.gadget = gadget := by
  obtain ⟨layout, hlookup, _, _, hwidth, heq⟩ := w.gadgetRun
  have hl : layout = DiamondBackend.layout0 := by
    simpa [DiamondBackend.backend] using hlookup.symm
  subst layout
  simpa only [castMatrixColumns, gadget] using heq

/-- The actual common-one preimage ignores the terminal row's second component because
    its target has a zero second row. Its error comes from the same state and preimage. -/
theorem initial_one_encoding (params : Stage_encrypt.Params)
    {decoder key one digits half publicInputs publicCircuit}
    (w : FinalPublicWitness DiamondBackend.backend params decoder key one digits half
      publicInputs publicCircuit)
    (secret : ExactMatrix q n 1 1) (selector : ExactMatrix q n 1 2)
    (state : ExactMatrix q n 1 inner) (B P : Nat)
    (hsecret : selector 0 0 = secret 0 0)
    (hstate : Approx state (selector * w.base) B) (hpreimage : PreimageWithin one P) :
    BooleanEncodingWithin secret (publicInputs 0) 1 (state * one) (projection * B * P) := by
  have hp := final_state_project (by decide : 0 < n) hstate hpreimage
  have htarget : selector * w.oneTarget = secret * publicInputs 0 - secret * gadget := by
    rw [final_selector_rows selector _ _ _ w.oneRows secret
      (fun _ _ ↦ selector 0 1) hsecret.symm rfl]
    simp only [Matrix.mul_zero, add_zero, Matrix.mul_sub, initial_registered_gadget params w]
  simpa only [BooleanEncodingWithin, Matrix.mul_assoc, w.oneEquation, htarget, one_smul]
    using hp

/-- The witness slot consumes its actual terminal row and sampled preimage. The second
    row is -G, so the row's secret-times-bit coefficient becomes the BGG payload term. -/
theorem initial_witness_encoding (B P : Nat)
    (secret : ExactMatrix q n 1 1) (selector : ExactMatrix q n 1 2)
    (state : ExactMatrix q n 1 inner) (base : ExactMatrix q n 2 inner)
    (publicKey output : ExactMatrix q n 1 ell) (target : ExactMatrix q n 2 ell)
    (preimage : ExactMatrix q n inner ell) (bit : Bool)
    (hsecret : selector 0 0 = secret 0 0)
    (hbit : selector 0 1 = secret 0 0 * (if bit then 1 else 0))
    (hstate : Approx state (selector * base) B)
    (hrows : concatRows publicKey (-gadget) target)
    (hpreimage : base * preimage = target) (hbounded : PreimageWithin preimage P)
    (hrun : output = state * preimage) :
    BooleanEncodingWithin secret publicKey (if bit then 1 else 0) output (projection * B * P) := by
  have hp := final_state_project (by decide : 0 < n) hstate hbounded
  have htarget : selector * target =
      secret * publicKey - (if bit then (1 : ExactPoly q n) else 0) • (secret * gadget) := by
    rw [final_selector_rows selector _ _ _ hrows secret
      (fun _ _ ↦ selector 0 1) hsecret.symm rfl]
    funext i j
    have hi : i = 0 := Subsingleton.elim _ _
    subst i
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, Matrix.mul_apply,
      Fin.sum_univ_one, Matrix.neg_apply, hbit, smul_eq_mul]
    ring
  have ho : output = state * preimage := hrun
  simpa only [BooleanEncodingWithin, ← ho, Matrix.mul_assoc, hpreimage, htarget] using hp


theorem initial_select_value {α : Type} (flag : Int) (left right output : α)
    (h : MxxRuntime.select flag [left, right] output) :
    output = if flag = 0 then left else right := by
  obtain ⟨position, hposition, houtput⟩ := h
  change Fin 2 at position
  fin_cases position
  · have hf : flag = 0 := hposition.symm
    simpa only [hf, ↓reduceIte, List.get] using houtput
  · have hf : flag = 1 := hposition.symm
    simpa only [hf, Int.one_ne_zero, ↓reduceIte, List.get] using houtput

/-- The fused initialization run exposes its shared address and aligned selectors. -/
structure InitialLaneFacts (lane : Nat) (start last bit : Int)
    (zeroCipher oneCipher zeroKey oneKey : ExactMatrix q n 1 ell)
    (zeroMessage oneMessage : ExactMatrix q n 1 1)
    (cipher keys : Fin witnessSlots → ExactMatrix q n 1 ell)
    (messages : Fin witnessSlots → ExactMatrix q n 1 1)
    (output : ExactMatrix q n 1 ell × ExactMatrix q n 1 ell ×
      ExactMatrix q n 1 1 × Unit) where
  index : Int
  c : ExactMatrix q n 1 ell
  k : ExactMatrix q n 1 ell
  m : ExactMatrix q n 1 1
  address : index = if start ≤ (lane : Int) ∧ (lane : Int) ≤ last
    then (lane : Int) - start else 0
  cipherGet : familyGetDynamic cipher index c
  keyGet : familyGetDynamic keys index k
  messageGet : familyGetDynamic messages index m
  cipherEquation : output.1 = if (lane : Int) ≤ start - 1
    then (if bit = 0 then zeroCipher else oneCipher)
    else (if start ≤ (lane : Int) ∧ (lane : Int) ≤ last then c else zeroCipher)
  keyEquation : output.2.1 = if (lane : Int) ≤ start - 1
    then (if bit = 0 then zeroKey else oneKey)
    else (if start ≤ (lane : Int) ∧ (lane : Int) ≤ last then k else zeroKey)
  messageEquation : output.2.2.1 = if (lane : Int) ≤ start - 1
    then (if bit = 0 then zeroMessage else oneMessage)
    else (if start ≤ (lane : Int) ∧ (lane : Int) ≤ last then m else zeroMessage)

theorem generated_initial_lane_facts (backend : BackendContext) (params : Stage_decrypt.Params)
    (lane : Nat) (start last bit : Int)
    (zc oc zk ok : ExactMatrix q n 1 ell) (zm om : ExactMatrix q n 1 1)
    (cs ks : Fin witnessSlots → ExactMatrix q n 1 ell)
    (ms : Fin witnessSlots → ExactMatrix q n 1 1) (output)
    (h : Stage_decrypt.parallel_generatedRoot_28 backend params lane
      (start, last, zc, cs, bit, oc, zk, ks, ok, zm, ms, om, ()) output) :
    Nonempty (InitialLaneFacts lane start last bit zc oc zk ok zm om cs ks ms output) := by
  dsimp only [Stage_decrypt.parallel_generatedRoot_28] at h
  obtain ⟨index, c, wc, ic, outc, k, wk, ik, outk, m, wm, im, outm, h⟩ := h
  dsimp only [Stage_decrypt.parallel_generatedRoot_28.constraints_0,
    Stage_decrypt.parallel_generatedRoot_28.constraints_1] at h
  have hcg : familyGetDynamic cs index c := by tauto
  have hkg : familyGetDynamic ks index k := by tauto
  have hmg : familyGetDynamic ms index m := by tauto
  have hout : output = (outc, outk, outm, ()) := by tauto
  have hi := initial_select_value _ _ _ _ (show select _ [0 + 0, (lane : Int) - start] index from by tauto)
  have hwc := initial_select_value _ _ _ _ (show select _ [zc, c] wc from by tauto)
  have hic := initial_select_value _ _ _ _ (show select bit [zc, oc] ic from by tauto)
  have hoc := initial_select_value _ _ _ _ (show select _ [wc, ic] outc from by tauto)
  have hwk := initial_select_value _ _ _ _ (show select _ [zk, k] wk from by tauto)
  have hik := initial_select_value _ _ _ _ (show select bit [zk, ok] ik from by tauto)
  have hok := initial_select_value _ _ _ _ (show select _ [wk, ik] outk from by tauto)
  have hwm := initial_select_value _ _ _ _ (show select _ [zm, m] wm from by tauto)
  have him := initial_select_value _ _ _ _ (show select bit [zm, om] im from by tauto)
  have hom := initial_select_value _ _ _ _ (show select _ [wm, im] outm from by tauto)
  clear h
  refine ⟨⟨index, c, k, m, ?_, hcg, hkg, hmg, ?_, ?_, ?_⟩⟩ <;>
    by_cases hstart : start ≤ (lane : Int) <;>
    by_cases hend : (lane : Int) ≤ last <;> simp_all <;> split_ifs <;> first | rfl | omega

/-- Ciphertext, key, and plaintext use the same address and branch at initialization. -/
theorem generated_initial_lane_within (backend : BackendContext)
    (params : Stage_decrypt.Params) (lane B : Nat) (start last bit : Int)
    (secret : ExactMatrix q n 1 1) (oneCipher oneKey : ExactMatrix q n 1 ell)
    (cipher keys : Fin witnessSlots → ExactMatrix q n 1 ell)
    (messages : Fin witnessSlots → ExactMatrix q n 1 1) (output)
    (hone : BooleanEncodingWithin secret oneKey 1 oneCipher B)
    (hfamily : ∀ slot, BooleanEncodingWithin secret (keys slot)
      (messages slot 0 0) (cipher slot) B ∧
      (messages slot 0 0 = 0 ∨ messages slot 0 0 = 1))
    (hrun : Stage_decrypt.parallel_generatedRoot_28 backend params lane
      (start, last, 0, cipher, bit, oneCipher, 0, keys, oneKey, 0, messages, 1, ()) output) :
    BooleanEncodingWithin secret output.2.1 (output.2.2.1 0 0) output.1 B ∧
      (output.2.2.1 0 0 = 0 ∨ output.2.2.1 0 0 = 1) := by
  obtain ⟨w⟩ := generated_initial_lane_facts backend params lane start last bit
    0 oneCipher 0 oneKey 0 1 cipher keys messages output hrun
  obtain ⟨slot, hslot, hc, hk⟩ := circuit_gather_same_index cipher keys w.index w.c w.k
    w.cipherGet w.keyGet
  obtain ⟨slot', hslot', _, hm⟩ := circuit_gather_same_index cipher messages w.index w.c w.m
    w.cipherGet w.messageGet
  have heq : slot' = slot := Fin.ext (by omega)
  subst slot'
  rw [w.cipherEquation, w.keyEquation, w.messageEquation]
  have hz := boolean_encoding_zero secret B
  by_cases hleft : (lane : Int) ≤ start - 1
  · by_cases hb : bit = 0 <;> simp only [hleft, hb, ↓reduceIte, Matrix.zero_apply,
      Matrix.one_apply_eq]
    · exact ⟨hz, Or.inl trivial⟩
    · exact ⟨hone, Or.inr trivial⟩
  · by_cases hw : start ≤ (lane : Int) ∧ (lane : Int) ≤ last <;>
      simp only [hleft, hw, ↓reduceIte, Matrix.zero_apply]
    · simpa only [hc, hk, hm] using hfamily slot
    · exact ⟨hz, Or.inl trivial⟩


/-- The current reference initializer and decoder initializer read the same raw slot. -/
theorem generated_initial_plaintext_lane_agrees
    (params : Requirement_2.Params) (lane : Nat) (start width bit : Int)
    (zc oc zk ok : ExactMatrix q n 1 ell)
    (cipher keys : Fin witnessSlots → ExactMatrix q n 1 ell)
    (messages : Fin witnessSlots → ExactMatrix q n 1 1)
    (raw : Fin circuitWidth → Int) (output) (reference : Bool)
    (w : InitialLaneFacts lane start (start + width - 1) bit
      zc oc zk ok 0 1 cipher keys messages output)
    (hbit : bit = 0 ∨ bit = 1)
    (hmessages : ∀ slot : Fin witnessSlots,
      messages slot 0 0 = if raw ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩ = 1
        then 1 else 0)
    (hr : Requirement_2.parallel_generatedRoot_17 params lane
      (start, start + width, raw, bit, ()) reference) :
    output.2.2.1 0 0 = if reference then 1 else 0 := by
  dsimp only [Requirement_2.parallel_generatedRoot_17] at hr
  obtain ⟨sample, witnessChoice, choice, hr⟩ := hr
  dsimp only [Requirement_2.parallel_generatedRoot_17.constraints_0] at hr
  have hget : familyGetDynamic raw
      (((if decide ((lane : Int) ≤ start + width - 1) then 1 else 0) -
        (if decide ((lane : Int) ≤ start - 1) then 1 else 0)) * ((lane : Int) - start)) sample := by tauto
  have hw := initial_select_value _ _ _ _ (show select _ [0 + 0, sample] witnessChoice from by tauto)
  have hi := initial_select_value _ _ _ _ (show select _ [witnessChoice, bit] choice from by tauto)
  have hout : reference = decide (choice = 1) := by tauto
  clear hr
  rw [w.messageEquation]
  by_cases hleft : (lane : Int) ≤ start - 1
  · rcases hbit with hb | hb <;> simp_all
  · have hstart : start ≤ (lane : Int) := by omega
    by_cases hend : (lane : Int) ≤ start + width - 1
    · have hindex : w.index = (lane : Int) - start := by simp [w.address, hstart, hend]
      obtain ⟨slot, hslot, hm⟩ := w.messageGet
      obtain ⟨rawSlot, hrawSlot, hsample⟩ := hget
      have hposition : rawSlot = ⟨slot.val, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val < 3; omega⟩ := by
        apply Fin.ext
        change rawSlot.val = slot.val
        simp only [hleft, hend, decide_true, decide_false, Bool.false_eq_true,
          ↓reduceIte, sub_zero, one_mul] at hrawSlot
        omega
      have hscalar := hmessages slot
      rw [← hposition, ← hsample] at hscalar
      simp_all
    · simp_all; split_ifs <;> first | rfl | omega

/-- Encryption and decryption initialize each public key through the same raw branch. -/
theorem generated_initial_public_lane_agrees
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (lane : Nat) (start last bit index : Int)
    (zc oc : ExactMatrix q n 1 ell)
    (publicInputs : Fin stateCount → ExactMatrix q n 1 ell)
    (cipher keys : Fin witnessSlots → ExactMatrix q n 1 ell)
    (messages : Fin witnessSlots → ExactMatrix q n 1 1) (output)
    (publicOutput : ExactMatrix q n 1 ell)
    (w : InitialLaneFacts lane start last bit zc oc 0 (publicInputs 0) 0 1
      cipher keys messages output)
    (hkeys : ∀ slot : Fin witnessSlots, keys slot = publicInputs
      ⟨slot.val + 1, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val + 1 < 2; omega⟩)
    (hi : Stage_encrypt.parallel_generatedRoot_20 backend hashModel params lane
      (start, last, ()) index)
    (hr : Stage_encrypt.parallel_generatedRoot_22 backend hashModel params lane
      (start, last, 0, publicInputs, index, bit, publicInputs 0, ()) publicOutput) :
    output.2.1 = publicOutput := by
  dsimp only [Stage_encrypt.parallel_generatedRoot_20] at hi
  obtain ⟨selectedIndex, _, _, _, hselectIndex, hindexOutput⟩ := hi
  have hindex := initial_select_value _ _ _ _ hselectIndex
  dsimp only [Stage_encrypt.parallel_generatedRoot_22] at hr
  obtain ⟨key, witnessChoice, instanceChoice, result, hr⟩ := hr
  dsimp only [Stage_encrypt.parallel_generatedRoot_22.constraints_0] at hr
  have hget : familyGetDynamic publicInputs index key := by tauto
  have hw := initial_select_value _ _ _ _ (show select _ [0, key] witnessChoice from by tauto)
  have hb := initial_select_value _ _ _ _ (show select bit [0, publicInputs 0] instanceChoice from by tauto)
  have ho := initial_select_value _ _ _ _ (show select _ [witnessChoice, instanceChoice] result from by tauto)
  have hout : publicOutput = result := by tauto
  clear hr
  rw [w.keyEquation]
  by_cases hstart : start ≤ (lane : Int)
  · have hleft : ¬(lane : Int) ≤ start - 1 := by omega
    by_cases hend : (lane : Int) ≤ last
    · have hshift : index = w.index + 1 := by simp_all [w.address]
      obtain ⟨slot, hslot, hk⟩ := w.keyGet
      have hpub : familyGetDynamic publicInputs index (keys slot) := by
        refine ⟨⟨slot.val + 1, by have hs := slot.isLt; change slot.val < 1 at hs; change slot.val + 1 < 2; omega⟩, ?_, hkeys slot⟩
        simp only [Nat.cast_add, Nat.cast_one]
        omega
      have heq : w.k = key := hk.trans (circuit_lookup_unique hpub hget)
      simp_all
    · simp_all; split_ifs <;> first | rfl | omega
  · have hleft : (lane : Int) ≤ start - 1 := by omega
    simp_all

#print axioms initial_one_encoding
#print axioms initial_witness_encoding
#print axioms generated_initial_lane_facts
#print axioms generated_initial_lane_within
#print axioms generated_initial_plaintext_lane_agrees
#print axioms generated_initial_public_lane_agrees

end DiamondGeneratedProof
