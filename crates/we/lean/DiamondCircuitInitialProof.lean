import DiamondProofParameters
import DiamondCircuitRequirementProof
import DiamondFinalEncodingProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

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
theorem initial_witness_encoding (params : Stage_decrypt.Params) (slot B P : Nat)
    (secret : ExactMatrix q n 1 1) (selector : ExactMatrix q n 1 2)
    (state : ExactMatrix q n 1 inner) (base : ExactMatrix q n 2 inner)
    (publicKey output : ExactMatrix q n 1 ell) (target : ExactMatrix q n 2 ell)
    (preimage : ExactMatrix q n inner ell) (bit : Bool)
    (hsecret : selector 0 0 = secret 0 0)
    (hbit : selector 0 1 = secret 0 0 * (if bit then 1 else 0))
    (hstate : Approx state (selector * base) B)
    (hrows : concatRows publicKey (-gadget) target)
    (hpreimage : base * preimage = target) (hbounded : PreimageWithin preimage P)
    (hrun : Stage_decrypt.parallel_generatedRoot_31 DiamondBackend.backend params slot
      (state, preimage, ()) output) :
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

/-- The three real initialization selectors use one signed flag for ciphertext, key,
    and message. This also applies definitionally to subsequent initialization selects. -/
theorem initial_triple_select_within (backend : BackendContext)
    (params : Stage_decrypt.Params) (lane B : Nat) (flag : Int)
    (secret : ExactMatrix q n 1 1)
    (leftCipher rightCipher leftKey rightKey output outputKey : ExactMatrix q n 1 ell)
    (leftMessage rightMessage outputMessage : ExactMatrix q n 1 1)
    (hl : BooleanEncodingWithin secret leftKey (leftMessage 0 0) leftCipher B)
    (hr : BooleanEncodingWithin secret rightKey (rightMessage 0 0) rightCipher B)
    (hlBool : leftMessage 0 0 = 0 ∨ leftMessage 0 0 = 1)
    (hrBool : rightMessage 0 0 = 0 ∨ rightMessage 0 0 = 1)
    (hcipher : Stage_decrypt.parallel_generatedRoot_33 backend params lane
      (flag, leftCipher, rightCipher, ()) output)
    (hkey : Stage_decrypt.parallel_generatedRoot_46 backend params lane
      (flag, leftKey, rightKey, ()) outputKey)
    (hmessage : Stage_decrypt.parallel_generatedRoot_58 backend params lane
      (flag, leftMessage, rightMessage, ()) outputMessage) :
    BooleanEncodingWithin secret outputKey (outputMessage 0 0) output B ∧
      (outputMessage 0 0 = 0 ∨ outputMessage 0 0 = 1) := by
  obtain ⟨cipher, _, _, _, ⟨ci, hci, hcv⟩, hco⟩ := hcipher
  obtain ⟨key, _, _, _, ⟨ki, hki, hkv⟩, hko⟩ := hkey
  obtain ⟨message, _, _, _, ⟨mi, hmi, hmv⟩, hmo⟩ := hmessage
  change Fin 2 at ci ki mi
  have hki' : ki = ci := Fin.ext (by dsimp at hki hci; omega)
  have hmi' : mi = ci := Fin.ext (by dsimp at hmi hci; omega)
  subst ki
  subst mi
  have hc := hco.trans hcv
  have hk := hko.trans hkv
  have hm := hmo.trans hmv
  fin_cases ci
  · simpa only [List.get, hc, hk, hm] using And.intro hl hlBool
  · simpa only [List.get, hc, hk, hm] using And.intro hr hrBool

/-- Actual initialization candidates and all three aligned selection stages yield the
    initial circuit lane invariant. No equality of the final arrays is assumed. -/
theorem generated_initial_lane_within (backend : BackendContext)
    (params : Stage_decrypt.Params) (lane B : Nat) (instanceFlag witnessFlag instanceBit : Int)
    (secret : ExactMatrix q n 1 1)
    (oneCipher oneKey witnessCipher witnessKey witnessChoiceCipher witnessChoiceKey
      instanceChoiceCipher instanceChoiceKey output outputKey : ExactMatrix q n 1 ell)
    (witnessMessage witnessChoiceMessage instanceChoiceMessage outputMessage :
      ExactMatrix q n 1 1)
    (hone : BooleanEncodingWithin secret oneKey 1 oneCipher B)
    (hwitness : BooleanEncodingWithin secret witnessKey (witnessMessage 0 0) witnessCipher B)
    (hwitnessBool : witnessMessage 0 0 = 0 ∨ witnessMessage 0 0 = 1)
    (hwc : Stage_decrypt.parallel_generatedRoot_33 backend params lane
      (witnessFlag, oneCipher - oneCipher, witnessCipher, ()) witnessChoiceCipher)
    (hwk : Stage_decrypt.parallel_generatedRoot_46 backend params lane
      (witnessFlag, oneKey - oneKey, witnessKey, ()) witnessChoiceKey)
    (hwm : Stage_decrypt.parallel_generatedRoot_58 backend params lane
      (witnessFlag, 0, witnessMessage, ()) witnessChoiceMessage)
    (hic : Stage_decrypt.parallel_generatedRoot_37 backend params lane
      (instanceBit, oneCipher - oneCipher, oneCipher, ()) instanceChoiceCipher)
    (hik : Stage_decrypt.parallel_generatedRoot_49 backend params lane
      (instanceBit, oneKey - oneKey, oneKey, ()) instanceChoiceKey)
    (him : Stage_decrypt.parallel_generatedRoot_61 backend params lane
      (instanceBit, 0, 1, ()) instanceChoiceMessage)
    (hoc : Stage_decrypt.parallel_generatedRoot_38 backend params lane
      (instanceFlag, witnessChoiceCipher, instanceChoiceCipher, ()) output)
    (hok : Stage_decrypt.parallel_generatedRoot_50 backend params lane
      (instanceFlag, witnessChoiceKey, instanceChoiceKey, ()) outputKey)
    (hom : Stage_decrypt.parallel_generatedRoot_62 backend params lane
      (instanceFlag, witnessChoiceMessage, instanceChoiceMessage, ()) outputMessage) :
    BooleanEncodingWithin secret outputKey (outputMessage 0 0) output B ∧
      (outputMessage 0 0 = 0 ∨ outputMessage 0 0 = 1) := by
  have hz : BooleanEncodingWithin secret (oneKey - oneKey) 0 (oneCipher - oneCipher) B := by
    simpa only [sub_self] using boolean_encoding_zero secret B
  have hmz : (0 : ExactMatrix q n 1 1) 0 0 = 0 ∨
      (0 : ExactMatrix q n 1 1) 0 0 = 1 := Or.inl rfl
  have hmo : (1 : ExactMatrix q n 1 1) 0 0 = 0 ∨
      (1 : ExactMatrix q n 1 1) 0 0 = 1 := Or.inr (by simp)
  obtain ⟨hw, hwb⟩ := initial_triple_select_within backend params lane B witnessFlag secret
    _ _ _ _ _ _ _ _ _ hz hwitness hmz hwitnessBool hwc hwk hwm
  obtain ⟨hi, hib⟩ := initial_triple_select_within backend params lane B instanceBit secret
    _ _ _ _ _ _ _ _ _ hz (by simpa using hone) hmz hmo hic hik him
  exact initial_triple_select_within backend params lane B instanceFlag secret
    _ _ _ _ _ _ _ _ _ hw hi hwb hib hoc hok hom

/-- Actual dynamic gathers retain the same slot across all three initial families. -/
theorem initial_gathered_triple_within (backend : BackendContext)
    (params : Stage_decrypt.Params) (lane B : Nat) (index : Int)
    (secret : ExactMatrix q n 1 1)
    (cipher keys : Fin witnessSlots → ExactMatrix q n 1 ell)
    (messages : Fin witnessSlots → ExactMatrix q n 1 1)
    (output outputKey : ExactMatrix q n 1 ell)
    (outputMessage : ExactMatrix q n 1 1)
    (hfamily : ∀ slot, BooleanEncodingWithin secret (keys slot)
      (messages slot 0 0) (cipher slot) B ∧
      (messages slot 0 0 = 0 ∨ messages slot 0 0 = 1))
    (hc : Stage_decrypt.parallel_generatedRoot_32 backend params lane
      (index, cipher, ()) output)
    (hk : Stage_decrypt.parallel_generatedRoot_45 backend params lane
      (index, keys, ()) outputKey)
    (hm : Stage_decrypt.parallel_generatedRoot_57 backend params lane
      (index, messages, ()) outputMessage) :
    BooleanEncodingWithin secret outputKey (outputMessage 0 0) output B ∧
      (outputMessage 0 0 = 0 ∨ outputMessage 0 0 = 1) := by
  obtain ⟨c, _, _, hc, hco⟩ := hc
  obtain ⟨k, _, _, hk, hko⟩ := hk
  obtain ⟨m, _, _, hm, hmo⟩ := hm
  obtain ⟨slot, hslot, hcv, hkv⟩ := circuit_gather_same_index cipher keys index c k hc hk
  obtain ⟨slot', hslot', _, hmv⟩ := circuit_gather_same_index cipher messages index c m hc hm
  have heq : slot' = slot := Fin.ext (by omega)
  subst slot'
  simpa only [hco, hko, hmo, hcv, hkv, hmv] using hfamily slot

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

/-- The actual plaintext selectors agree with the requirement's initial lane, including
    instance priority and padding. The selected witness scalar is linked separately by
    the actual common-index gathers, not by an equality of final initial arrays. -/
theorem generated_initial_plaintext_lane_agrees (backend : BackendContext)
    (params : Stage_decrypt.Params) (referenceParams : Requirement_2.Params)
    (lane : Nat) (instanceWidth witnessWidth instanceFlag witnessFlag instanceBit witnessBit : Int)
    (witnessMessage witnessChoice instanceChoice output : ExactMatrix q n 1 1)
    (reference : Bool)
    (hinstanceBit : instanceBit = 0 ∨ instanceBit = 1)
    (hwitness : witnessMessage 0 0 = if witnessBit = 1 then 1 else 0)
    (hinstanceFlag : Stage_decrypt.parallel_generatedRoot_19 backend params lane
      instanceWidth instanceFlag)
    (hwitnessFlag : Stage_decrypt.parallel_generatedRoot_24 backend params lane
      (instanceWidth, instanceWidth + witnessWidth - 1, ()) witnessFlag)
    (hw : Stage_decrypt.parallel_generatedRoot_58 backend params lane
      (witnessFlag, 0, witnessMessage, ()) witnessChoice)
    (hi : Stage_decrypt.parallel_generatedRoot_61 backend params lane
      (instanceBit, 0, 1, ()) instanceChoice)
    (ho : Stage_decrypt.parallel_generatedRoot_62 backend params lane
      (instanceFlag, witnessChoice, instanceChoice, ()) output)
    (hr : Requirement_2.parallel_generatedRoot_22 referenceParams lane
      (instanceBit, witnessBit, instanceWidth, instanceWidth + witnessWidth, ()) reference) :
    output 0 0 = if reference then 1 else 0 := by
  obtain ⟨w, _, _, _, hw, hwo⟩ := hw
  obtain ⟨i, _, _, _, hi, hio⟩ := hi
  obtain ⟨o, _, _, _, ho, hoo⟩ := ho
  have hwv := initial_select_value _ _ _ _ hw
  have hiv := initial_select_value _ _ _ _ hi
  have hov := initial_select_value _ _ _ _ ho
  dsimp only [Stage_decrypt.parallel_generatedRoot_19] at hinstanceFlag
  dsimp only [Stage_decrypt.parallel_generatedRoot_24] at hwitnessFlag
  dsimp only [Requirement_2.parallel_generatedRoot_22] at hr
  obtain ⟨r, r', _, _, _, hr, _, _, _, hr', hrout⟩ := hr
  have hrv := initial_select_value _ _ _ _ hr
  have hrv' := initial_select_value _ _ _ _ hr'
  by_cases hleft : (lane : Int) ≤ instanceWidth - 1
  · rcases hinstanceBit with hbit | hbit <;>
      simp_all
  · have hstart : instanceWidth ≤ (lane : Int) := by omega
    by_cases hend : (lane : Int) ≤ instanceWidth + witnessWidth - 1
    · simp_all
    · simp_all

/-- The generated clamped witness addresses equal the requirement's addresses. -/
theorem generated_initial_indices_agree (backend : BackendContext)
    (params : Stage_decrypt.Params) (referenceParams : Requirement_2.Params)
    (lane : Nat) (instanceWidth witnessWidth instanceBit index referenceIndex : Int)
    (hwidth : 0 ≤ witnessWidth)
    (hd : Stage_decrypt.parallel_generatedRoot_27 backend params lane
      (instanceWidth, instanceWidth + witnessWidth - 1, ()) index)
    (hr : Requirement_2.parallel_generatedRoot_20 referenceParams lane
      (instanceBit, instanceWidth + witnessWidth, instanceWidth, ()) referenceIndex) :
    index = referenceIndex := by
  dsimp only [Stage_decrypt.parallel_generatedRoot_27] at hd
  obtain ⟨selected, _, _, _, hs, ho⟩ := hd
  have hv := initial_select_value _ _ _ _ hs
  dsimp only [Requirement_2.parallel_generatedRoot_20] at hr
  by_cases hleft : (lane : Int) ≤ instanceWidth - 1
  · have hstart : ¬instanceWidth ≤ (lane : Int) := by omega
    have hend : (lane : Int) ≤ instanceWidth + witnessWidth - 1 := by omega
    simp_all
  · have hstart : instanceWidth ≤ (lane : Int) := by omega
    by_cases hend : (lane : Int) ≤ instanceWidth + witnessWidth - 1 <;> simp_all

/-- Actual raw-witness extraction, bit selection, and the later message gather agree
    with the requirement's raw-witness gather at their proven common address. -/
theorem generated_witness_plaintext_agrees (backend : BackendContext)
    (params : Stage_decrypt.Params) (referenceParams : Requirement_2.Params)
    (lane : Nat) (index referenceIndex selectedWitness : Int)
    (raw : Fin circuitWidth → Int) (slotIndices sampled : Fin witnessSlots → Int)
    (messages : Fin witnessSlots → ExactMatrix q n 1 1)
    (selectedMessage : ExactMatrix q n 1 1)
    (hindex : index = referenceIndex)
    (hindices : ∀ slot : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_3 backend params slot
      () (slotIndices slot))
    (hsampled : ∀ slot : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_5 backend params slot
      (slotIndices slot, raw, ()) (sampled slot))
    (hbits : ∀ slot : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_56 backend params slot
      (sampled slot, 0, 1, ()) (messages slot))
    (hmessage : Stage_decrypt.parallel_generatedRoot_57 backend params lane
      (index, messages, ()) selectedMessage)
    (hreference : Requirement_2.parallel_generatedRoot_21 referenceParams lane
      (referenceIndex, raw, ()) selectedWitness) :
    selectedMessage 0 0 = if selectedWitness = 1 then 1 else 0 := by
  obtain ⟨message, _, _, ⟨slot, hslot, hmessage⟩, houtput⟩ := hmessage
  have hslotIndex : slotIndices slot = referenceIndex := by
    have hi := hindices slot
    change slotIndices slot = (slot.val : Int) + 0 at hi
    simpa only [add_zero, hslot, hindex] using hi
  obtain ⟨sample, _, _, hs, hso⟩ := hsampled slot
  obtain ⟨referenceValue, _, _, hr, hro⟩ := hreference
  rw [hslotIndex] at hs
  have hvalue : sampled slot = selectedWitness :=
    hso.trans ((circuit_lookup_unique hs hr).trans hro.symm)
  obtain ⟨bitMessage, _, _, _, ⟨position, hposition, hbit⟩, hbo⟩ := hbits slot
  change Fin 2 at position
  have hout : selectedMessage = bitMessage := houtput.trans (hmessage.trans hbo)
  fin_cases position
  · have hv : selectedWitness = 0 := hvalue.symm.trans hposition.symm
    simp only [hout, hbit, List.get, Matrix.zero_apply, hv, Int.zero_ne_one, ↓reduceIte]
  · have hv : selectedWitness = 1 := hvalue.symm.trans hposition.symm
    simp only [hout, hbit, List.get, Matrix.one_apply_eq, hv, ↓reduceIte]

/-- Whole-family initial plaintext agreement obtained from actual initialization scopes
    on the same raw instance/witness arrays. No final-family equality is a hypothesis. -/
theorem generated_initial_plaintext_agrees (backend : BackendContext)
    (params : Stage_decrypt.Params) (referenceParams : Requirement_2.Params)
    (instanceWidth witnessWidth : Int) (initial : CircuitState)
    (rawInstance rawWitness instanceFlags witnessFlags indices referenceIndices
      referenceWitnesses : Fin circuitWidth → Int)
    (slotIndices sampled : Fin witnessSlots → Int)
    (slotMessages : Fin witnessSlots → ExactMatrix q n 1 1)
    (witnessMessages witnessChoices instanceChoices : Fin circuitWidth → ExactMatrix q n 1 1)
    (referenceInitial : Fin circuitWidth → Bool)
    (hwidth : 0 ≤ witnessWidth)
    (hraw : ∀ lane, rawInstance lane = 0 ∨ rawInstance lane = 1)
    (hslotIndices : ∀ slot : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_3 backend params slot
      () (slotIndices slot))
    (hsampled : ∀ slot : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_5 backend params slot
      (slotIndices slot, rawWitness, ()) (sampled slot))
    (hbits : ∀ slot : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_56 backend params slot
      (sampled slot, 0, 1, ()) (slotMessages slot))
    (hindices : ∀ lane : Fin circuitWidth, Stage_decrypt.parallel_generatedRoot_27 backend params lane
      (instanceWidth, instanceWidth + witnessWidth - 1, ()) (indices lane))
    (hreferenceIndices : ∀ lane : Fin circuitWidth,
      Requirement_2.parallel_generatedRoot_20 referenceParams lane
        (rawInstance lane, instanceWidth + witnessWidth, instanceWidth, ())
        (referenceIndices lane))
    (hwitnesses : ∀ lane : Fin circuitWidth, Stage_decrypt.parallel_generatedRoot_57 backend params lane
      (indices lane, slotMessages, ()) (witnessMessages lane))
    (hreferenceWitnesses : ∀ lane : Fin circuitWidth,
      Requirement_2.parallel_generatedRoot_21 referenceParams lane
        (referenceIndices lane, rawWitness, ()) (referenceWitnesses lane))
    (hinstanceFlags : ∀ lane : Fin circuitWidth, Stage_decrypt.parallel_generatedRoot_19 backend params lane
      instanceWidth (instanceFlags lane))
    (hwitnessFlags : ∀ lane : Fin circuitWidth, Stage_decrypt.parallel_generatedRoot_24 backend params lane
      (instanceWidth, instanceWidth + witnessWidth - 1, ()) (witnessFlags lane))
    (hw : ∀ lane : Fin circuitWidth, Stage_decrypt.parallel_generatedRoot_58 backend params lane
      (witnessFlags lane, 0, witnessMessages lane, ()) (witnessChoices lane))
    (hi : ∀ lane : Fin circuitWidth, Stage_decrypt.parallel_generatedRoot_61 backend params lane
      (rawInstance lane, 0, 1, ()) (instanceChoices lane))
    (ho : ∀ lane : Fin circuitWidth, Stage_decrypt.parallel_generatedRoot_62 backend params lane
      (instanceFlags lane, witnessChoices lane, instanceChoices lane, ()) (initial.2.2.1 lane))
    (hr : ∀ lane : Fin circuitWidth, Requirement_2.parallel_generatedRoot_22 referenceParams lane
      (rawInstance lane, referenceWitnesses lane, instanceWidth,
        instanceWidth + witnessWidth, ()) (referenceInitial lane)) :
    CircuitPlaintextAgrees initial referenceInitial := by
  intro lane
  have hindex := generated_initial_indices_agree backend params referenceParams lane
    instanceWidth witnessWidth (rawInstance lane) (indices lane) (referenceIndices lane)
    hwidth (hindices lane) (hreferenceIndices lane)
  have hwitness := generated_witness_plaintext_agrees backend params referenceParams lane
    (indices lane) (referenceIndices lane) (referenceWitnesses lane) rawWitness slotIndices
    sampled slotMessages (witnessMessages lane) hindex hslotIndices hsampled hbits
    (hwitnesses lane) (hreferenceWitnesses lane)
  exact generated_initial_plaintext_lane_agrees backend params referenceParams lane
    instanceWidth witnessWidth (instanceFlags lane) (witnessFlags lane) (rawInstance lane)
    (referenceWitnesses lane) (witnessMessages lane) (witnessChoices lane)
    (instanceChoices lane) (initial.2.2.1 lane) (referenceInitial lane) (hraw lane) hwitness
    (hinstanceFlags lane) (hwitnessFlags lane) (hw lane) (hi lane) (ho lane) (hr lane)

#print axioms initial_one_encoding
#print axioms initial_witness_encoding
#print axioms generated_initial_lane_within
#print axioms initial_gathered_triple_within
#print axioms generated_initial_plaintext_lane_agrees
#print axioms generated_initial_indices_agree
#print axioms generated_witness_plaintext_agrees
#print axioms generated_initial_plaintext_agrees

end DiamondGeneratedProof
