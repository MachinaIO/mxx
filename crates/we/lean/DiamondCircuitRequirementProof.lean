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

theorem circuit_select_at {α : Type} {values : List α} (index : Nat) (output : α)
    (hrun : MxxRuntime.select (index : Int) values output) (hindex : index < values.length) :
    output = values.get ⟨index, hindex⟩ := by
  obtain ⟨position, hposition, hvalue⟩ := hrun
  have heq : position = ⟨index, hindex⟩ := Fin.ext (by change position.val = index; omega)
  exact heq ▸ hvalue

theorem circuit_mask_value {α : Type} (lane : Nat) (active : Int) (zero selected output : α)
    (hrun : MxxRuntime.select (if decide ((lane : Int) ≤ active - 1) then 1 else 0)
      [zero, selected] output) : output = if (lane : Int) < active then selected else zero := by
  by_cases ha : (lane : Int) < active
  · have hf : decide ((lane : Int) ≤ active - 1) = true := by apply decide_eq_true; omega
    rw [hf, if_pos rfl] at hrun
    exact (circuit_select_at 1 output hrun (by simp)).trans (by simp [ha])
  · have hf : decide ((lane : Int) ≤ active - 1) = false := by apply decide_eq_false; omega
    rw [hf] at hrun
    exact (circuit_select_at 0 output hrun (by simp)).trans (by simp [ha])

/-- The equation uses the same shared reads, selectors, and mask as the fused lane. -/
theorem generated_circuit_plaintext_equation
    (params : Stage_decrypt.Params) (layer : Nat) (current next : CircuitState)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (honeMessage : oneMessage 0 0 = 1)
    (hrun : Stage_decrypt.sequential_generatedRoot_33 DiamondBackend.backend params layer
      (current.1, current.2.1, current.2.2.1, activeCounts, oneCipher, kinds, leftSources,
        rightSources, onePublic, oneMessage, ()) next) :
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
  obtain ⟨⟨active, oc, op, om⟩, _, _, hactive, _, hlanes, hout⟩ := hrun
  refine ⟨active, hactive, ?_⟩
  intro lane
  obtain ⟨kind, left, right, digits, sc, sp, sm, hk, hl, hr, _, _, _, hm, _, _, hmm⟩ :=
    generated_circuit_lane_facts DiamondBackend.backend params layer lane active current
      kinds leftSources rightSources oneCipher onePublic oneMessage _ _ _ (hlanes lane)
  refine ⟨kind, left, right, hk, hl, hr, ?_⟩
  rw [hout]
  change om lane 0 0 = _
  have hs := circuit_select_at kind.val sm hm (by simp [circuitMessageCandidates])
  have hmask := circuit_mask_value lane active (matrixSub oneMessage oneMessage) sm (om lane) hmm
  rw [hmask]
  split_ifs with ha
  · rw [hs]
    fin_cases kind <;> simp [circuitMessageCandidates, matrixSub, matrixAdd,
      matrixMulScalarLeft, Matrix.mul_apply, matrixPolynomial, honeMessage, List.get]
    ring
  · simp [matrixSub]

/-- The requirement lane reads identical addresses and applies the six Boolean gates. -/
theorem generated_requirement_lane_equation (params : Requirement_2.Params) (layer lane : Nat)
    (active : Int) (reference : Fin circuitWidth → Bool)
    (kinds leftSources rightSources : Fin metadataCount → Int) (output : Bool)
    (hrun : Requirement_2.parallel_sequential_generatedRoot_22_7 params layer lane
      (active, kinds, (layer : Int), reference, leftSources, rightSources, ()) output) :
    ∃ (kind : Fin 6) (left right : Fin circuitWidth),
      familyGetDynamic kinds ((layer : Int) * params.max_layer_width + lane) (kind.val : Int) ∧
      familyGetDynamic leftSources ((layer : Int) * params.max_layer_width + lane) (left.val : Int) ∧
      familyGetDynamic rightSources ((layer : Int) * params.max_layer_width + lane) (right.val : Int) ∧
      (if output then (1 : ExactPoly q n) else 0) = if (lane : Int) < active then
        [0, 1, if reference left then 1 else 0, 1 - (if reference left then 1 else 0),
          (if reference left then 1 else 0) * (if reference right then 1 else 0),
          (if reference left then 1 else 0) + (if reference right then 1 else 0) -
            2 * ((if reference left then 1 else 0) * (if reference right then 1 else 0))].get kind
        else 0 := by
  dsimp only [Requirement_2.parallel_sequential_generatedRoot_22_7] at hrun
  obtain ⟨ki, li, lv, ri, rv, selected, masked, h⟩ := hrun
  dsimp only [Requirement_2.parallel_sequential_generatedRoot_22_7.constraints_0] at h
  rcases h with ⟨_, _, hk, _, _, hl, _, _, hlv, _, _, hr, _, _, hrv,
    hkn, hklt, _, hs, _, _, _, hm, hout⟩
  obtain ⟨left, hleft, hleftValue⟩ := hlv
  obtain ⟨right, hright, hrightValue⟩ := hrv
  let kind : Fin 6 := ⟨ki.toNat, by omega⟩
  have hkind : (kind.val : Int) = ki := by dsimp [kind]; omega
  refine ⟨kind, left, right, ?_, ?_, ?_, ?_⟩
  · simpa only [hkind] using hk
  · simpa only [hleft] using hl
  · simpa only [hright] using hr
  · rw [← hkind] at hs
    have hs' := circuit_select_at kind.val selected hs kind.isLt
    have hm' := circuit_mask_value lane active false selected masked hm
    rw [hout, hm']
    by_cases ha : (lane : Int) < active
    · simp only [if_pos ha]
      rw [hs', ← hleftValue, ← hrightValue]
      clear_value kind
      cases lv <;> cases rv <;> fin_cases kind <;> norm_num [List.get]
    · simp only [if_neg ha, Bool.false_eq_true, ↓reduceIte]

/-- Paired actual layers preserve plaintext agreement without assuming an accepting result. -/
theorem generated_circuit_requirement_layer_agrees
    (params : Stage_decrypt.Params) (requirementParams : Requirement_2.Params) (layer : Nat)
    (current next : CircuitState) (reference referenceNext : Fin circuitWidth → Bool)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (hwidth : params.max_layer_width = requirementParams.max_layer_width)
    (honeMessage : oneMessage 0 0 = 1) (hagrees : CircuitPlaintextAgrees current reference)
    (hcircuit : Stage_decrypt.sequential_generatedRoot_33 DiamondBackend.backend params layer
      (current.1, current.2.1, current.2.2.1, activeCounts, oneCipher, kinds, leftSources,
        rightSources, onePublic, oneMessage, ()) next)
    (hrequirement : Requirement_2.sequential_generatedRoot_22 requirementParams layer
      (reference, activeCounts, kinds, leftSources, rightSources, ()) referenceNext) :
    CircuitPlaintextAgrees next referenceNext := by
  obtain ⟨active, hactive, hc⟩ := generated_circuit_plaintext_equation params layer
    current next activeCounts kinds leftSources rightSources oneCipher onePublic oneMessage
    honeMessage hcircuit
  obtain ⟨⟨referenceActive, values⟩, _, _, hreferenceActive, _, hlanes, hout⟩ := hrequirement
  have ha : referenceActive = active := circuit_lookup_unique hreferenceActive hactive
  rw [hout]
  intro lane
  obtain ⟨kind, left, right, hk, hl, hr, heq⟩ := hc lane
  obtain ⟨rkind, rleft, rright, hrk, hrl, hrr, hreq⟩ :=
    generated_requirement_lane_equation requirementParams layer lane referenceActive reference
      kinds leftSources rightSources _ (hlanes lane)
  rw [← hwidth] at hrk hrl hrr
  have hkval := circuit_lookup_unique hk hrk
  have hlval := circuit_lookup_unique hl hrl
  have hrval := circuit_lookup_unique hr hrr
  have hkpos : rkind = kind := Fin.ext (by omega)
  have hlpos : rleft = left := Fin.ext (by omega)
  have hrpos : rright = right := Fin.ext (by omega)
  subst rkind; subst rleft; subst rright
  change next.2.2.1 lane 0 0 = if values lane then 1 else 0
  rw [heq, hagrees left, hagrees right]
  simpa only [ha] using hreq.symm

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
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_33
        DiamondBackend.backend params layer
        (current.1, current.2.1, current.2.2.1, activeCounts, oneCipher, kinds, leftSources, rightSources,
          onePublic, oneMessage, ()) next) count initial output)
    (hrequirement : MxxIR.IterRuns
      (fun layer current next ↦ Requirement_2.sequential_generatedRoot_22 requirementParams layer
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


theorem generated_accepting_requirement_plaintext
    (params : Stage_decrypt.Params) (requirementParams : Requirement_2.Params)
    (instanceValues witnessValues : Fin circuitWidth → Int) (activeCounts : Fin circuitDepth → Int)
    (kinds leftSources rightSources : Fin metadataCount → Int) (outputSources : Fin 1 → Int)
    (hwidth : params.max_layer_width = requirementParams.max_layer_width)
    (hroot : Requirement_2.generatedRoot requirementParams
      (instanceValues, witnessValues, activeCounts, kinds, leftSources, rightSources,
        outputSources, ()) true) :
    ∃ (referenceInitial : Fin circuitWidth → Bool) (position : Fin circuitWidth),
      (∀ lane : Fin circuitWidth, Requirement_2.parallel_generatedRoot_17 requirementParams lane
        (requirementParams.instance_width,
          requirementParams.instance_width + requirementParams.witness_width,
          witnessValues, instanceValues lane, ()) (referenceInitial lane)) ∧
      (position.val : Int) = outputSources 0 ∧
      ∀ (initial output : CircuitState) (oneCipher onePublic : ExactMatrix q n 1 ell)
        (oneMessage : ExactMatrix q n 1 1),
        oneMessage 0 0 = 1 → CircuitPlaintextAgrees initial referenceInitial →
        MxxIR.IterRuns
          (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_33
            DiamondBackend.backend params layer
            (current.1, current.2.1, current.2.2.1, activeCounts, oneCipher, kinds, leftSources,
              rightSources, onePublic, oneMessage, ()) next)
          requirementParams.depth.toNat initial output →
        output.2.2.1 position 0 0 = 1 := by
  dsimp only [Requirement_2.generatedRoot] at hroot
  obtain ⟨rootWitness, h⟩ := hroot
  rcases rootWitness with ⟨instanceChecks, witnessChecks, inputChecks, validInputs,
    referenceInitial, referenceOutput, selectedIndex, accepted⟩
  dsimp only [Requirement_2.generatedRoot.body, Requirement_2.generatedRoot.constraints_0] at h
  rcases h with ⟨_, _, _, _, _, _, _, _, _, hinitial,
    _, hloop, hsource, _, _, houtput, hsuccess⟩
  have haccept : accepted = true := by
    cases accepted
    · cases validInputs <;> norm_num at hsuccess
    · rfl
  obtain ⟨sourcePosition, _, hsourceValue⟩ := hsource
  have hsourcePosition : sourcePosition = 0 := Subsingleton.elim _ _
  have hselectedIndex : selectedIndex = outputSources 0 := by
    simpa only [hsourcePosition] using hsourceValue
  obtain ⟨position, hposition, hvalue⟩ := houtput
  have hposition' : (position.val : Int) = outputSources 0 := hposition.trans hselectedIndex
  refine ⟨referenceInitial, position,
    by simpa only [add_zero] using hinitial, hposition', ?_⟩
  intro initial output oneCipher onePublic oneMessage hone hagrees hrun
  have hfinal := generated_circuit_requirement_iteration_agrees params requirementParams
    requirementParams.depth.toNat initial output referenceInitial referenceOutput
    activeCounts kinds leftSources rightSources oneCipher onePublic oneMessage
    hwidth hone hagrees hrun hloop
  have hreference : referenceOutput position = true := hvalue.symm.trans haccept
  simpa only [hreference, ↓reduceIte] using hfinal position


#print axioms generated_circuit_plaintext_equation
#print axioms generated_requirement_lane_equation
#print axioms generated_circuit_requirement_layer_agrees
#print axioms generated_circuit_requirement_iteration_agrees
#print axioms generated_accepting_requirement_plaintext

end DiamondGeneratedProof
