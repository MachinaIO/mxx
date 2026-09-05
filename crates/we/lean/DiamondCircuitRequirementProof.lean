import DiamondProofParameters
import DiamondCircuitLayerProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxRecDepth 8192
set_option maxHeartbeats 1600000

def CircuitPlaintextAgrees (state : CircuitState) (values : Fin circuitWidth → Bool) : Prop :=
  ∀ lane, state.2.2.1 lane 0 0 = if values lane then 1 else 0

theorem circuit_lookup_unique {α : Type} {N : Nat}
    {values : Fin N → α} {index : Int} {left right : α}
    (hl : familyGetDynamic values index left) (hr : familyGetDynamic values index right) :
    left = right := by
  obtain ⟨position, _, hl, hr⟩ := circuit_gather_same_index values values index left right hl hr
  exact hl.trans hr.symm

/-- A scalar output equation extracted from the actual circuit layer, retaining the exact
    metadata and message gather positions. It is not a second circuit evaluator. -/
theorem generated_circuit_plaintext_equation
    (params : Stage_decrypt.Params) (layer : Nat) (current next : CircuitState)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (honeMessage : oneMessage 0 0 = 1)
    (hrun : Stage_decrypt.sequential_generatedRoot_67 DiamondBackend.backend params layer
      (current.1, current.2.1, current.2.2.1, activeCounts, kinds, leftSources, rightSources,
        oneCipher, onePublic, oneMessage, ()) next) :
    ∃ active : Int, familyGetDynamic activeCounts (layer : Int) active ∧
      ∀ lane : Fin circuitWidth, ∃ (kind : Fin 6) (left right : Fin circuitWidth),
        familyGetDynamic kinds ((layer : Int) * params.max_layer_width + lane.val)
          (kind.val : Int) ∧
        familyGetDynamic leftSources ((layer : Int) * params.max_layer_width + lane.val)
          (left.val : Int) ∧
        familyGetDynamic rightSources ((layer : Int) * params.max_layer_width + lane.val)
          (right.val : Int) ∧
        next.2.2.1 lane 0 0 = if (lane.val : Int) < active then
          [0, 1, current.2.2.1 left 0 0, 1 - current.2.2.1 left 0 0,
            current.2.2.1 left 0 0 * current.2.2.1 right 0 0,
            current.2.2.1 left 0 0 + current.2.2.1 right 0 0 -
              2 * (current.2.2.1 left 0 0 * current.2.2.1 right 0 0)].get kind else 0 := by
  dsimp only [Stage_decrypt.sequential_generatedRoot_67] at hrun
  rcases hrun with ⟨active, flags, w5, w6, addresses, gateKinds, leftIndices, w13, w14,
    rightIndices, w18, digits, w20, w21, w23, w24, w25, w26, w28, w29, w30, w31,
    w33, w34, w35, w36, w37, w38, w39, w40, w41, w42, w44, w45, w46, w47,
    w48, w49, w50, w51, w52, w53, h⟩
  rcases h with ⟨_, _, hactive, _, h3, _, _, _, _, _, h7, _, h9, _, h11, _, _,
    _, _, _, h16, _, _, _, _, _, _, _, _, _, h23, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, h44, _, h45, _, h46,
    _, h47, _, h48, _, h49, _, h50, _, h51, _, h52, _, h53, hout⟩
  refine ⟨active, hactive, ?_⟩
  intro lane
  obtain ⟨selected, hn, hlt, _, ⟨kind, hkind, hselected⟩, hsout⟩ := h52 lane
  change Fin 6 at kind
  dsimp only at hselected
  obtain ⟨leftMessage, _, _, ⟨left, hleft, hlval⟩, hlout⟩ := h23 lane
  obtain ⟨rightMessage, _, _, ⟨right, hright, hrval⟩, hrout⟩ := h47 lane
  have haddr : addresses lane = (layer : Int) * params.max_layer_width + lane.val := by
    simpa [Stage_decrypt.parallel_sequential_generatedRoot_67_7] using h7 lane
  have hk : familyGetDynamic kinds ((layer : Int) * params.max_layer_width + lane.val)
      (kind.val : Int) := by
    obtain ⟨value, _, _, hget, hvalue⟩ := h9 lane
    have heq : (kind.val : Int) = value := hkind.trans hvalue
    simpa only [haddr, heq] using hget
  have hl : familyGetDynamic leftSources ((layer : Int) * params.max_layer_width + lane.val)
      (left.val : Int) := by
    obtain ⟨value, _, _, hget, hvalue⟩ := h11 lane
    have heq : (left.val : Int) = value := hleft.trans hvalue
    simpa only [haddr, heq] using hget
  have hr : familyGetDynamic rightSources ((layer : Int) * params.max_layer_width + lane.val)
      (right.val : Int) := by
    obtain ⟨value, _, _, hget, hvalue⟩ := h16 lane
    have heq : (right.val : Int) = value := hright.trans hvalue
    simpa only [haddr, heq] using hget
  refine ⟨kind, left, right, hk, hl, hr, ?_⟩
  have hm0 : w45 lane = 0 := by
    have h : w45 lane = w44 lane - w44 lane := h45 lane
    simpa using h
  have hm1 : w44 lane 0 0 = 1 := by rw [h44 lane]; exact honeMessage
  have hm3 : w46 lane 0 0 = 1 - w23 lane 0 0 := by
    rw [h46 lane]
    change w44 lane 0 0 - w23 lane 0 0 = _
    rw [hm1]
  have hm4 : w48 lane 0 0 = w23 lane 0 0 * w47 lane 0 0 := by rw [h48 lane]; rfl
  have hm5 : w51 lane 0 0 = w23 lane 0 0 + w47 lane 0 0 -
      2 * (w23 lane 0 0 * w47 lane 0 0) := by
    rw [h51 lane, h49 lane, h50 lane, h48 lane]
    change w23 lane 0 0 + w47 lane 0 0 - ((w23 lane 0 0 * w47 lane 0 0) *
      (matrixPolynomial [2] : ExactMatrix q n 1 1) 0 0) = _
    have htwo : (matrixPolynomial [2] : ExactMatrix q n 1 1) 0 0 = 2 := by
      simp [matrixPolynomial]
    rw [htwo]
    ring
  have hs : w52 lane 0 0 =
      [0, 1, current.2.2.1 left 0 0, 1 - current.2.2.1 left 0 0,
        current.2.2.1 left 0 0 * current.2.2.1 right 0 0,
        current.2.2.1 left 0 0 + current.2.2.1 right 0 0 -
          2 * (current.2.2.1 left 0 0 * current.2.2.1 right 0 0)].get kind := by
    rw [hsout.trans hselected]
    fin_cases kind <;>
      simp only [List.get, hm0, hm1, hm3, hm4, hm5, hlout, hlval, hrout, hrval,
        Matrix.zero_apply]
  rw [hout]
  change w53 lane 0 0 = _
  obtain ⟨masked, _, _, _, ⟨position, hposition, hmasked⟩, hmout⟩ := h53 lane
  have hflag := h3 lane
  dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_67_3] at hflag
  by_cases ha : (lane.val : Int) < active
  · have hf : decide (Int.ofNat lane.val ≤ active - 1) = true := by
      apply decide_eq_true
      change (lane.val : Int) ≤ active - 1
      omega
    rw [hf] at hflag
    have hp : position = (⟨1, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition hflag ⊢
      omega
    subst position
    rw [hmout.trans hmasked, if_pos ha]
    exact hs
  · have hf : decide (Int.ofNat lane.val ≤ active - 1) = false := by
      apply decide_eq_false
      change ¬ (lane.val : Int) ≤ active - 1
      omega
    rw [hf] at hflag
    have hp : position = (⟨0, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition hflag ⊢
      omega
    subst position
    rw [hmout.trans hmasked, if_neg ha, hm0]
    rfl

/-- The generated requirement and ciphertext layers agree on every plaintext when their
    actual input families agree and they read the same circuit metadata. -/
theorem generated_circuit_requirement_layer_agrees
    (params : Stage_decrypt.Params) (requirementParams : Requirement_2.Params) (layer : Nat)
    (current next : CircuitState) (reference referenceNext : Fin circuitWidth → Bool)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (hwidth : params.max_layer_width = requirementParams.max_layer_width)
    (honeMessage : oneMessage 0 0 = 1) (hagrees : CircuitPlaintextAgrees current reference)
    (hcircuit : Stage_decrypt.sequential_generatedRoot_67 DiamondBackend.backend params layer
      (current.1, current.2.1, current.2.2.1, activeCounts, kinds, leftSources, rightSources,
        oneCipher, onePublic, oneMessage, ()) next)
    (hrequirement : Requirement_2.sequential_generatedRoot_27 requirementParams layer
      (reference, activeCounts, kinds, leftSources, rightSources, ()) referenceNext) :
    CircuitPlaintextAgrees next referenceNext := by
  obtain ⟨active, hactive, hc⟩ := generated_circuit_plaintext_equation params layer
    current next activeCounts kinds leftSources rightSources oneCipher onePublic oneMessage
    honeMessage hcircuit
  dsimp only [Requirement_2.sequential_generatedRoot_27] at hrequirement
  obtain ⟨addresses, gateKinds, leftIndices, leftValues, rightIndices, rightValues,
    referenceActive, values, _, haddr, _, hkinds, _, hlefts, _, hleftValues, _, hrights,
    _, hrightValues, _, _, hreferenceActive, _, hgates, hout⟩ := hrequirement
  have ha : referenceActive = active := circuit_lookup_unique hreferenceActive hactive
  rw [hout]
  intro lane
  obtain ⟨kind, left, right, hkind, hleft, hright, hvalue⟩ := hc lane
  have haddress : addresses lane = (layer : Int) * params.max_layer_width + lane.val := by
    simpa [Requirement_2.parallel_sequential_generatedRoot_27_0, hwidth] using haddr lane
  obtain ⟨kindValue, _, _, hkindLookup, hkindOut⟩ := hkinds lane
  have hkind' : (kind.val : Int) = gateKinds lane := by
    have hk : (kind.val : Int) = kindValue :=
      circuit_lookup_unique hkind (by simpa only [haddress] using hkindLookup)
    exact hk.trans hkindOut.symm
  obtain ⟨leftIndex, _, _, hleftLookup, hleftOut⟩ := hlefts lane
  have hleft' : (left.val : Int) = leftIndices lane := by
    have hl : (left.val : Int) = leftIndex :=
      circuit_lookup_unique hleft (by simpa only [haddress] using hleftLookup)
    exact hl.trans hleftOut.symm
  obtain ⟨rightIndex, _, _, hrightLookup, hrightOut⟩ := hrights lane
  have hright' : (right.val : Int) = rightIndices lane := by
    have hr : (right.val : Int) = rightIndex :=
      circuit_lookup_unique hright (by simpa only [haddress] using hrightLookup)
    exact hr.trans hrightOut.symm
  obtain ⟨leftValue, _, _, ⟨leftPosition, hleftPosition, hleftValue⟩, hlOut⟩ :=
    hleftValues lane
  have hlp : leftPosition = left := Fin.ext (by dsimp at hleftPosition; omega)
  have hlv : leftValues lane = reference left := by
    simpa only [hlp] using hlOut.trans hleftValue
  obtain ⟨rightValue, _, _, ⟨rightPosition, hrightPosition, hrightValue⟩, hrOut⟩ :=
    hrightValues lane
  have hrp : rightPosition = right := Fin.ext (by dsimp at hrightPosition; omega)
  have hrv : rightValues lane = reference right := by
    simpa only [hrp] using hrOut.trans hrightValue
  have hgate := generated_requirement_gate_value requirementParams layer lane kind
    (leftValues lane) (rightValues lane) (values lane) referenceActive
    (hkind'.symm ▸ hgates lane)
  change next.2.2.1 lane 0 0 = if values lane then 1 else 0
  rw [hvalue]
  rw [ha, hlv, hrv] at hgate
  rw [hagrees left, hagrees right]
  exact hgate.symm

/-- Paired induction over the two actual loop derivations; no lane or layer expansion. -/
theorem generated_circuit_requirement_iteration_agrees
    (params : Stage_decrypt.Params) (requirementParams : Requirement_2.Params) (count : Nat)
    (initial output : CircuitState) (referenceInitial referenceOutput : Fin circuitWidth → Bool)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (hwidth : params.max_layer_width = requirementParams.max_layer_width)
    (honeMessage : oneMessage 0 0 = 1)
    (hagrees : CircuitPlaintextAgrees initial referenceInitial)
    (hcircuit : MxxIR.IterRuns
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_67
        DiamondBackend.backend params layer
        (current.1, current.2.1, current.2.2.1, activeCounts, kinds, leftSources, rightSources,
          oneCipher, onePublic, oneMessage, ()) next) count initial output)
    (hrequirement : MxxIR.IterRuns
      (fun layer current next ↦ Requirement_2.sequential_generatedRoot_27 requirementParams layer
        (current, activeCounts, kinds, leftSources, rightSources, ()) next)
      count referenceInitial referenceOutput) :
    CircuitPlaintextAgrees output referenceOutput := by
  induction hcircuit generalizing referenceOutput with
  | zero =>
      cases hrequirement
      exact hagrees
  | @step count initial current next hprevious hstep ih =>
      obtain ⟨referenceCurrent, hreferencePrevious, hreferenceStep⟩ :=
        MxxIR.IterRuns.step_of_succ hrequirement
      exact generated_circuit_requirement_layer_agrees params requirementParams count
        current next referenceCurrent referenceOutput activeCounts kinds leftSources rightSources
        oneCipher onePublic oneMessage hwidth honeMessage
        (ih referenceCurrent hagrees hreferencePrevious)
        hstep hreferenceStep

/-- A successful actual requirement root supplies its own initial family, loop execution,
    and accepting output position. The remaining implication concerns initial plaintext
    alignment only; neither final acceptance nor a noise bound is a caller premise. -/
theorem generated_accepting_requirement_plaintext
    (params : Stage_decrypt.Params) (requirementParams : Requirement_2.Params)
    (instanceValues witnessValues : Fin circuitWidth → Int) (activeCounts : Fin circuitDepth → Int)
    (kinds leftSources rightSources : Fin metadataCount → Int) (outputSources : Fin 1 → Int)
    (hwidth : params.max_layer_width = requirementParams.max_layer_width)
    (hroot : Requirement_2.generatedRoot requirementParams
      (instanceValues, witnessValues, activeCounts, kinds, leftSources, rightSources,
        outputSources, ()) true) :
    ∃ (witnessIndices selectedWitnesses : Fin circuitWidth → Int) (referenceInitial : Fin circuitWidth → Bool)
      (position : Fin circuitWidth),
      (∀ lane : Fin circuitWidth, Requirement_2.parallel_generatedRoot_20 requirementParams lane
        (instanceValues lane, requirementParams.instance_width + requirementParams.witness_width,
          requirementParams.instance_width, ()) (witnessIndices lane)) ∧
      (∀ lane : Fin circuitWidth, Requirement_2.parallel_generatedRoot_21 requirementParams lane
        (witnessIndices lane, witnessValues, ()) (selectedWitnesses lane)) ∧
      (∀ lane : Fin circuitWidth, Requirement_2.parallel_generatedRoot_22 requirementParams lane
        (instanceValues lane, selectedWitnesses lane, requirementParams.instance_width,
          requirementParams.instance_width + requirementParams.witness_width, ())
        (referenceInitial lane)) ∧
      (position.val : Int) = outputSources 0 ∧
      ∀ (initial output : CircuitState) (oneCipher onePublic : ExactMatrix q n 1 ell)
        (oneMessage : ExactMatrix q n 1 1),
        oneMessage 0 0 = 1 → CircuitPlaintextAgrees initial referenceInitial →
        MxxIR.IterRuns
          (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_67
            DiamondBackend.backend params layer
            (current.1, current.2.1, current.2.2.1, activeCounts, kinds, leftSources,
              rightSources, oneCipher, onePublic, oneMessage, ()) next)
          requirementParams.depth.toNat initial output →
        output.2.2.1 position 0 0 = 1 := by
  dsimp only [Requirement_2.generatedRoot] at hroot
  obtain ⟨instanceChecks, witnessChecks, inputChecks, validInputs, witnessIndices,
    selectedWitnesses, referenceInitial, referenceOutput, selectedIndex, accepted, h⟩ := hroot
  rcases h with ⟨_, _, _, _, _, _, _, _, _, hindices, _, hwitnesses, _, hinitial,
    _, hloop, _, _, hsource, _, _, houtput, hsuccess⟩
  have haccept : accepted = true := by
    cases accepted
    · simp at hsuccess
    · rfl
  obtain ⟨sourcePosition, _, hsourceValue⟩ := hsource
  have hsourcePosition : sourcePosition = 0 := Subsingleton.elim _ _
  have hselectedIndex : selectedIndex = outputSources 0 := by
    simpa only [hsourcePosition] using hsourceValue
  obtain ⟨position, hposition, hvalue⟩ := houtput
  have hposition' : (position.val : Int) = outputSources 0 := hposition.trans hselectedIndex
  refine ⟨witnessIndices, selectedWitnesses, referenceInitial, position,
    by simpa only [add_zero] using hindices, hwitnesses,
    by simpa only [add_zero] using hinitial, hposition', ?_⟩
  intro initial output oneCipher onePublic oneMessage hone hagrees hrun
  have hfinal := generated_circuit_requirement_iteration_agrees params requirementParams
    requirementParams.depth.toNat initial output referenceInitial referenceOutput
    activeCounts kinds leftSources rightSources oneCipher onePublic oneMessage
    hwidth hone hagrees hrun hloop
  have hreference : referenceOutput position = true := hvalue.symm.trans haccept
  simpa only [hreference, ↓reduceIte] using hfinal position

#print axioms generated_circuit_plaintext_equation
#print axioms generated_circuit_requirement_layer_agrees
#print axioms generated_circuit_requirement_iteration_agrees
#print axioms generated_accepting_requirement_plaintext

end DiamondGeneratedProof
