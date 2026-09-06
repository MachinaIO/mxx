import DiamondProofParameters
import DiamondBooleanGateProof
import Requirement_2

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxRecDepth 8192
set_option maxHeartbeats 1600000

abbrev CircuitState := (Fin circuitWidth → ExactMatrix q n 1 ell) ×
  (Fin circuitWidth → ExactMatrix q n 1 ell) × (Fin circuitWidth → ExactMatrix q n 1 1) × Unit

/-- The invariant is pointwise in the actual three carried families. -/
def CircuitStateWithin (secret : ExactMatrix q n 1 1) (bound : Nat)
    (state : CircuitState) : Prop :=
  ∀ lane, BooleanEncodingWithin secret (state.2.1 lane) (state.2.2.1 lane 0 0)
    (state.1 lane) bound ∧
    (state.2.2.1 lane 0 0 = 0 ∨ state.2.2.1 lane 0 0 = 1)

theorem circuit_gather_same_index {α β : Type} {N : Nat}
    (xs : Fin N → α) (ys : Fin N → β) (index : Int) (x : α) (y : β)
    (hx : familyGetDynamic xs index x) (hy : familyGetDynamic ys index y) :
    ∃ position : Fin N, (position.val : Int) = index ∧ x = xs position ∧ y = ys position := by
  obtain ⟨ix, hix, hxx⟩ := hx
  obtain ⟨iy, hiy, hyy⟩ := hy
  have heq : ix = iy := by
    apply Fin.ext
    omega
  subst iy
  exact ⟨ix, hix, hxx, hyy⟩

/-- A complete generated Boolean layer preserves bounded encodings and Boolean plaintexts.
    Every candidate and gather below comes from this one execution of the actual layer. -/
theorem generated_circuit_layer_within
    (params : Stage_decrypt.Params) (layer B : Nat)
    (secret : ExactMatrix q n 1 1) (current next : CircuitState)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (hone : BooleanEncodingWithin secret onePublic 1 oneCipher B)
    (honeMessage : oneMessage 0 0 = 1)
    (hinvariant : CircuitStateWithin secret B current)
    (hrun : Stage_decrypt.sequential_generatedRoot_67 DiamondBackend.backend params layer
      (current.1, current.2.1, current.2.2.1, activeCounts, kinds, leftSources, rightSources,
        oneCipher, onePublic, oneMessage, ()) next) :
    CircuitStateWithin secret (factor * B) next := by
  dsimp only [Stage_decrypt.sequential_generatedRoot_67] at hrun
  rcases hrun with ⟨active, flags, w5, w6, addresses, gateKinds, leftIndices, w13, w14,
    rightIndices, w18, digits, w20, w21, w23, w24, w25, w26, w28, w29, w30, w31,
    w33, w34, w35, w36, w37, w38, w39, w40, w41, w42, w44, w45, w46, w47,
    w48, w49, w50, w51, w52, w53, h⟩
  rcases h with ⟨_, _, _, _, h3, _, h5, _, h6, _, _, _, _, _, _, _, h13,
    _, h14, _, _, _, h18, _, h19, _, h20, _, h21, _, h23, _, h24, _, h25,
    _, h26, _, h28, _, h29, _, h30, _, h31, _, h33, _, h34, _, h35, _, h36,
    _, h37, _, h38, _, h39, _, h40, _, h41, _, h42, _, h44, _, h45, _, h46,
    _, h47, _, h48, _, h49, _, h50, _, h51, _, h52, _, h53, hout⟩
  rw [hout]
  intro lane
  obtain ⟨lc, _, _, hleftCipher, hlc⟩ := h13 lane
  obtain ⟨lp, _, _, hleftPublic, hlp⟩ := h35 lane
  obtain ⟨lm, _, _, hleftMessage, hlm⟩ := h23 lane
  obtain ⟨leftPosition, hlIndex, hci, hpi⟩ := circuit_gather_same_index
    current.1 current.2.1 (leftIndices lane) lc lp hleftCipher hleftPublic
  obtain ⟨messagePosition, hmIndex, _, hmi⟩ := circuit_gather_same_index
    current.1 current.2.2.1 (leftIndices lane) lc lm hleftCipher hleftMessage
  have hpositions : leftPosition = messagePosition := Fin.ext (by omega)
  have hlWithin : BooleanEncodingWithin secret (w35 lane) (w23 lane 0 0) (w13 lane) B := by
    rw [hlc, hlp, hlm, hci, hpi, hmi, ← hpositions]
    exact (hinvariant leftPosition).1
  have hlBool : w23 lane 0 0 = 0 ∨ w23 lane 0 0 = 1 := by
    rw [hlm, hmi]
    exact (hinvariant messagePosition).2
  obtain ⟨rc, _, _, hrightCipher, hrc⟩ := h21 lane
  obtain ⟨rp, _, _, hrightPublic, hrp⟩ := h18 lane
  obtain ⟨rm, _, _, hrightMessage, hrm⟩ := h47 lane
  obtain ⟨ci, hciIndex, hciValue⟩ := hrightCipher
  obtain ⟨pi, hpiIndex, hpiValue⟩ := hrightPublic
  obtain ⟨mi, hmiIndex, hmiValue⟩ := hrightMessage
  have hip : pi = ci := Fin.ext (by omega)
  have him : mi = ci := Fin.ext (by omega)
  have hrWithin : BooleanEncodingWithin secret (w18 lane) (w47 lane 0 0) (w21 lane) B := by
    rw [hrc, hrp, hrm, hciValue, hpiValue, hmiValue, hip, him]
    exact (hinvariant ci).1
  have hrBool : w47 lane 0 0 = 0 ∨ w47 lane 0 0 = 1 := by
    rw [hrm, hmiValue]
    exact (hinvariant mi).2
  obtain ⟨bit, hbit⟩ : ∃ bit : Bool, w23 lane 0 0 = if bit then 1 else 0 := by
    rcases hlBool with hz | ho
    · exact ⟨false, hz⟩
    · exact ⟨true, ho⟩
  have hone' : BooleanEncodingWithin secret (w33 lane) 1 (w5 lane) B := by
    rw [h33 lane, h5 lane]
    exact hone
  have candidates := generated_boolean_candidates_within params layer lane B
    (w33 lane) (w35 lane) (w18 lane) (w5 lane) (w13 lane) (w21 lane) (w6 lane)
    (w14 lane) (w20 lane) (w24 lane) (w25 lane) (w26 lane) (w28 lane) (w29 lane)
    secret (w23 lane) (w47 lane 0 0) bit (digits lane) hbit hone' hlWithin hrWithin
    (h6 lane) (h14 lane) (h19 lane) (h20 lane) (h24 lane) (h25 lane)
    (h26 lane) (h28 lane) (h29 lane)
  have hk0 : w34 lane = 0 := by
    have h : w34 lane = w33 lane - w33 lane := h34 lane
    simpa using h
  have hm0 : w45 lane = 0 := by
    have h : w45 lane = w44 lane - w44 lane := h45 lane
    simpa using h
  have hm1 : w44 lane 0 0 = 1 := by rw [h44 lane]; exact honeMessage
  have hk3 : w36 lane = w33 lane - w35 lane := h36 lane
  have hk4 : w37 lane = w35 lane * digits lane := h37 lane
  have hk5 : w40 lane = w35 lane + w18 lane -
      (2 : ExactPoly q n) • (w35 lane * digits lane) := by
    rw [h40 lane, h38 lane, h39 lane, hk4]
    funext row column
    simp [matrixSub, matrixAdd, matrixMulScalarRight, matrixPolynomial,
      Matrix.smul_apply, mul_comm]
  have hm3 : w46 lane 0 0 = 1 - w23 lane 0 0 := by
    rw [h46 lane]
    change w44 lane 0 0 - w23 lane 0 0 = _
    rw [hm1]
  have hm4 : w48 lane 0 0 = w23 lane 0 0 * w47 lane 0 0 := by
    rw [h48 lane]
    rfl
  have hm5 : w51 lane 0 0 = w23 lane 0 0 + w47 lane 0 0 -
      2 * (w23 lane 0 0 * w47 lane 0 0) := by
    rw [h51 lane, h49 lane, h50 lane, h48 lane]
    change w23 lane 0 0 + w47 lane 0 0 - ((w23 lane 0 0 * w47 lane 0 0) *
      (matrixPolynomial [2] : ExactMatrix q n 1 1) 0 0) = _
    have htwo : (matrixPolynomial [2] : ExactMatrix q n 1 1) 0 0 = 2 := by
      simp [matrixPolynomial]
    rw [htwo]
    ring
  let cs : Fin 6 → ExactMatrix q n 1 ell :=
    fun k ↦ [w6 lane, w5 lane, w13 lane, w14 lane, w25 lane, w29 lane].get k
  let ks : Fin 6 → ExactMatrix q n 1 ell :=
    fun k ↦ [w34 lane, w33 lane, w35 lane, w36 lane, w37 lane, w40 lane].get k
  let ms : Fin 6 → ExactMatrix q n 1 1 :=
    fun k ↦ [w45 lane, w44 lane, w23 lane, w46 lane, w48 lane, w51 lane].get k
  have hcandidates : ∀ k, BooleanEncodingWithin secret (ks k) (ms k 0 0) (cs k)
      (factor * B) := by
    intro k
    have h := candidates k
    fin_cases k <;>
      simpa only [cs, ks, ms, List.get, hk0, hm0, hm1, hk3, hk4, hk5, hm3, hm4, hm5,
        Matrix.zero_apply] using h
  have hmessages : ∀ k, ms k 0 0 = 0 ∨ ms k 0 0 = 1 := by
    intro k
    have h := boolean_gate_message_closed _ _ hlBool hrBool k
    fin_cases k <;>
      simpa only [ms, List.get, hm0, hm1, hm3, hm4, hm5, Matrix.zero_apply] using h
  have hselect := h30 lane
  obtain ⟨chosen, hkNonneg, hkLt, _, hs, _⟩ := hselect
  let kind : Fin 6 := ⟨(gateKinds lane).toNat, by omega⟩
  have hkind : (kind.val : Int) = gateKinds lane := by dsimp [kind]; omega
  have hselected := generated_selected_encoding_within DiamondBackend.backend params layer lane
    (factor * B) kind secret cs ks ms (w30 lane) (w41 lane) (w52 lane)
    hcandidates (hkind.symm ▸ h30 lane) (hkind.symm ▸ h41 lane) (hkind.symm ▸ h52 lane)
  have hz := generated_encrypted_zero DiamondBackend.backend params layer lane
    (w5 lane) (w6 lane) (h6 lane)
  have hbounded := generated_masked_encoding_within DiamondBackend.backend params layer lane
    (factor * B) active (flags lane) secret (w30 lane) (w41 lane)
    (w31 lane) (w42 lane) (w52 lane) (w53 lane) hselected (h3 lane)
    (by simpa only [hz] using h31 lane) (by simpa only [hk0] using h42 lane)
    (by simpa only [hm0] using h53 lane)
  refine ⟨hbounded, ?_⟩
  change w53 lane 0 0 = 0 ∨ w53 lane 0 0 = 1
  obtain ⟨selectedMessage, _, _, _, ⟨position, hposition, hselectedMessage⟩, hmOut⟩ := h52 lane
  have hpositionKind : position = kind := by apply Fin.ext; dsimp at hposition; omega
  subst position
  have hmSelected : w52 lane = ms kind := hmOut.trans hselectedMessage
  obtain ⟨maskedMessage, _, _, _, ⟨position, _, hmaskedMessage⟩, hmOut⟩ := h53 lane
  fin_cases position
  · have heq : w53 lane = w45 lane := hmOut.trans hmaskedMessage
    exact Or.inl (by rw [heq, hm0]; rfl)
  · have heq : w53 lane = w52 lane := hmOut.trans hmaskedMessage
    rw [heq, hmSelected]
    exact hmessages kind

/-- The counted circuit run uses one symbolic layer induction, including zero iterations. -/
theorem generated_circuit_iteration_within
    (params : Stage_decrypt.Params) (count B : Nat)
    (secret : ExactMatrix q n 1 1) (initial output : CircuitState)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (hone : BooleanEncodingWithin secret onePublic 1 oneCipher B)
    (honeMessage : oneMessage 0 0 = 1)
    (hinitial : CircuitStateWithin secret B initial)
    (hrun : MxxIR.IterRuns
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_67
        DiamondBackend.backend params layer
        (current.1, current.2.1, current.2.2.1, activeCounts, kinds, leftSources, rightSources,
          oneCipher, onePublic, oneMessage, ()) next) count initial output) :
    CircuitStateWithin secret (factor ^ count * B) output := by
  apply MxxIR.IterRuns.invariant
    (Invariant := fun layer state ↦
      CircuitStateWithin secret (factor ^ layer * B) state)
    (by simpa using hinitial) _ hrun
  intro layer current next ih hstep
  have hbound : B ≤ factor ^ layer * B := by
    have hp : 1 ≤ factor ^ layer := Nat.one_le_pow _ _ (by unfold factor; omega)
    exact (one_mul B).symm.trans_le (Nat.mul_le_mul_right B hp)
  have h := generated_circuit_layer_within params layer
    (factor ^ layer * B) secret current next activeCounts kinds
    leftSources rightSources oneCipher onePublic oneMessage
    (boolean_encoding_mono hone hbound) honeMessage ih hstep
  convert h using 1
  rw [pow_succ]
  ring

/-- The actual Boolean requirement gate has the same scalar arithmetic as the generated
    ciphertext layer. This is local equivalence, not an assumed accepting output. -/
theorem generated_requirement_gate_value
    (params : Requirement_2.Params) (layer lane : Nat) (kind : Fin 6)
    (left right output : Bool) (active : Int)
    (hrun : Requirement_2.parallel_sequential_generatedRoot_27_13 params layer lane
      ((kind.val : Int), left, right, active, ()) output) :
    (if output then (1 : ExactPoly q n) else 0) =
      if (lane : Int) < active then
        ([0, 1, if left then 1 else 0, 1 - (if left then 1 else 0),
          (if left then 1 else 0) * (if right then 1 else 0),
          (if left then 1 else 0) + (if right then 1 else 0) -
            2 * ((if left then 1 else 0) * (if right then 1 else 0))].get kind)
      else 0 := by
  dsimp only [Requirement_2.parallel_sequential_generatedRoot_27_13] at hrun
  obtain ⟨selected, masked, _, _, _, hselect, _, _, _, hmask, hout⟩ := hrun
  obtain ⟨position, hposition, hselected⟩ := hselect
  have hp : position = kind := by apply Fin.ext; dsimp at hposition; omega
  subst position
  by_cases ha : (lane : Int) < active
  · have hf : decide (Int.ofNat lane ≤ active - 1) = true := by
      apply decide_eq_true
      change (lane : Int) ≤ active - 1
      omega
    rw [hf, if_pos rfl] at hmask
    obtain ⟨position, hposition, hmasked⟩ := hmask
    have hp : position = (⟨1, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    rw [if_pos ha, hout.trans (hmasked.trans hselected)]
    cases left <;> cases right <;> fin_cases kind <;> norm_num [List.get]
  · have hf : decide (Int.ofNat lane ≤ active - 1) = false := by
      apply decide_eq_false
      change ¬ (lane : Int) ≤ active - 1
      omega
    rw [hf] at hmask
    obtain ⟨position, hposition, hmasked⟩ := hmask
    have hp : position = (⟨0, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    rw [if_neg ha, hout.trans hmasked]
    rfl

#print axioms generated_circuit_layer_within
#print axioms generated_circuit_iteration_within
#print axioms generated_requirement_gate_value

end DiamondGeneratedProof
