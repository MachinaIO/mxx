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

/-- Candidate lists are the primitive arithmetic of the fused lane, with no new evaluation. -/
noncomputable def circuitCipherCandidates (one left right : ExactMatrix q n 1 ell)
    (message : ExactMatrix q n 1 1) (digits : ExactMatrix q n ell ell) :=
  let product := matrixAdd (matrixMul left digits) (matrixMulScalarRight right message)
  [matrixSub one one, one, left, matrixSub one left, product,
    matrixSub (matrixAdd left right)
      (matrixMulScalarRight product (matrixPolynomial [2] : ExactMatrix q n 1 1))]

noncomputable def circuitPublicCandidates (one left right : ExactMatrix q n 1 ell)
    (digits : ExactMatrix q n ell ell) :=
  [matrixSub one one, one, left, matrixSub one left, matrixMul left digits,
    matrixSub (matrixAdd left right)
      (matrixMulScalarRight (matrixMul left digits)
        (matrixPolynomial [2] : ExactMatrix q n 1 1))]

noncomputable def circuitMessageCandidates (one left right : ExactMatrix q n 1 1) :=
  [matrixSub one one, one, left, matrixSub one left, matrixMulScalarLeft left right,
    matrixSub (matrixAdd left right)
      (matrixMulScalarLeft (matrixMulScalarLeft left right)
        (matrixPolynomial [2] : ExactMatrix q n 1 1))]

/-- Extract the shared reads and selectors of one actual fused lane. -/
theorem generated_circuit_lane_facts (backend : BackendContext)
    (params : Stage_decrypt.Params) (layer lane : Nat) (active : Int)
    (current : CircuitState) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (outCipher outPublic : ExactMatrix q n 1 ell) (outMessage : ExactMatrix q n 1 1)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_33_12 backend params layer lane
      (active, oneCipher, kinds, (layer : Int), current.1, leftSources, current.2.1,
        rightSources, current.2.2.1, onePublic, oneMessage, ())
      (outCipher, outPublic, outMessage, ())) :
    ∃ (kind : Fin 6) (left right : Fin circuitWidth) (digits : ExactMatrix q n ell ell)
      (selectedCipher selectedPublic : ExactMatrix q n 1 ell)
      (selectedMessage : ExactMatrix q n 1 1),
      familyGetDynamic kinds ((layer : Int) * params.max_layer_width + lane) (kind.val : Int) ∧
      familyGetDynamic leftSources ((layer : Int) * params.max_layer_width + lane) (left.val : Int) ∧
      familyGetDynamic rightSources ((layer : Int) * params.max_layer_width + lane) (right.val : Int) ∧
      gadgetDecomposeRuns backend params.diamond_gadget_base params.diamond_digit_count
        (current.2.1 right) digits ∧
      MxxRuntime.select (kind.val : Int)
        (circuitCipherCandidates oneCipher (current.1 left) (current.1 right)
          (current.2.2.1 left) digits) selectedCipher ∧
      MxxRuntime.select (kind.val : Int)
        (circuitPublicCandidates onePublic (current.2.1 left) (current.2.1 right) digits)
        selectedPublic ∧
      MxxRuntime.select (kind.val : Int)
        (circuitMessageCandidates oneMessage (current.2.2.1 left) (current.2.2.1 right))
        selectedMessage ∧
      MxxRuntime.select (if decide ((lane : Int) ≤ active - 1) then 1 else 0)
        [matrixSub oneCipher oneCipher, selectedCipher] outCipher ∧
      MxxRuntime.select (if decide ((lane : Int) ≤ active - 1) then 1 else 0)
        [matrixSub onePublic onePublic, selectedPublic] outPublic ∧
      MxxRuntime.select (if decide ((lane : Int) ≤ active - 1) then 1 else 0)
        [matrixSub oneMessage oneMessage, selectedMessage] outMessage := by
  dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_33_12] at hrun
  rcases hrun with ⟨ki, li, lc, ri, rp, digits, rc, lm, sc, oc, lp, sp, op, rm, sm, om, h⟩
  dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_33_12.constraints_0,
    Stage_decrypt.parallel_sequential_generatedRoot_33_12.constraints_1] at h
  rcases h with ⟨_, _, hk, _, _, hl, _, _, hlc, _, _, hr, _, _, hrp, hd,
    _, _, hrc, _, _, hlm, hkn, hklt, _, hsc, _, _, _, hoc, _, _, hlp,
    _, _, _, hsp, _, _, _, hop, _, _, hrm, _, _, _, hsm, _, _, _, hom, hout⟩
  obtain ⟨left, hleft, hlc, hlp⟩ := circuit_gather_same_index current.1 current.2.1 li lc lp hlc hlp
  obtain ⟨leftM, hleftM, _, hlm⟩ := circuit_gather_same_index current.1 current.2.2.1 li lc lm
    ⟨left, hleft, hlc⟩ hlm
  have hlpos : leftM = left := Fin.ext (by omega)
  subst leftM
  obtain ⟨right, hright, hrc, hrp⟩ := circuit_gather_same_index current.1 current.2.1 ri rc rp hrc hrp
  obtain ⟨rightM, hrightM, _, hrm⟩ := circuit_gather_same_index current.1 current.2.2.1 ri rc rm
    ⟨right, hright, hrc⟩ hrm
  have hrpos : rightM = right := Fin.ext (by omega)
  subst rightM
  let kind : Fin 6 := ⟨ki.toNat, by omega⟩
  have hkind : (kind.val : Int) = ki := by dsimp [kind]; omega
  have hocout := congrArg Prod.fst hout
  have hopout := congrArg (fun x => x.2.1) hout
  have homout := congrArg (fun x => x.2.2.1) hout
  dsimp only at hocout hopout homout
  refine ⟨kind, left, right, digits, sc, sp, sm, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa only [hkind] using hk
  · simpa only [hleft] using hl
  · simpa only [hright] using hr
  · simpa only [hrp] using hd
  · simpa only [circuitCipherCandidates, hkind, hlc, hrc, hlm] using hsc
  · simpa only [circuitPublicCandidates, hkind, hlp, hrp] using hsp
  · simpa only [circuitMessageCandidates, hkind, hlm, hrm] using hsm
  · simpa only [hocout] using hoc
  · simpa only [hopout] using hop
  · simpa only [homout] using hom

/-- A complete fused Boolean layer preserves bounded encodings and Boolean plaintexts. -/
theorem generated_circuit_layer_within
    (params : Stage_decrypt.Params) (layer B : Nat)
    (secret : ExactMatrix q n 1 1) (current next : CircuitState)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (hone : BooleanEncodingWithin secret onePublic 1 oneCipher B)
    (honeMessage : oneMessage 0 0 = 1)
    (hinvariant : CircuitStateWithin secret B current)
    (hrun : Stage_decrypt.sequential_generatedRoot_33 DiamondBackend.backend params layer
      (current.1, current.2.1, current.2.2.1, activeCounts, oneCipher, kinds, leftSources,
        rightSources, onePublic, oneMessage, ()) next) :
    CircuitStateWithin secret (factor * B) next := by
  obtain ⟨⟨active, outCipher, outPublic, outMessage⟩, _, _, _, _, hlanes, hout⟩ := hrun
  rw [hout]
  intro lane
  obtain ⟨kind, left, right, digits, sc, sp, sm, _, _, _, hd, hc, hp, hm, hmc, hmp, hmm⟩ :=
    generated_circuit_lane_facts DiamondBackend.backend params layer lane active current
      kinds leftSources rightSources oneCipher onePublic oneMessage _ _ _ (hlanes lane)
  obtain ⟨hlWithin, hlBool⟩ := hinvariant left
  obtain ⟨hrWithin, hrBool⟩ := hinvariant right
  obtain ⟨bit, hbit⟩ : ∃ bit : Bool, current.2.2.1 left 0 0 = if bit then 1 else 0 := by
    rcases hlBool with hz | ho
    · exact ⟨false, hz⟩
    · exact ⟨true, ho⟩
  let lc := current.1 left
  let rc := current.1 right
  let lp := current.2.1 left
  let rp := current.2.1 right
  let lm := current.2.2.1 left
  let rm := current.2.2.1 right
  let product := matrixAdd (matrixMul lc digits) (matrixMulScalarRight rc lm)
  let double := matrixMulScalarRight product (matrixPolynomial [2] : ExactMatrix q n 1 1)
  have candidates := generated_boolean_candidates_within params B
    onePublic lp rp oneCipher lc rc (matrixSub oneCipher oneCipher)
    (matrixSub oneCipher lc) (matrixMul lc digits) (matrixMulScalarRight rc lm)
    product (matrixAdd lc rc) double (matrixSub (matrixAdd lc rc) double)
    secret lm (rm 0 0) bit digits hbit hone hlWithin hrWithin
    rfl rfl hd rfl rfl rfl rfl rfl rfl
  let cs : Fin 6 → ExactMatrix q n 1 ell := (circuitCipherCandidates oneCipher lc rc lm digits).get
  let ps : Fin 6 → ExactMatrix q n 1 ell := (circuitPublicCandidates onePublic lp rp digits).get
  let ms : Fin 6 → ExactMatrix q n 1 1 := (circuitMessageCandidates oneMessage lm rm).get
  have hmessage : ∀ k, ms k 0 0 =
      ([0, 1, lm 0 0, 1 - lm 0 0, lm 0 0 * rm 0 0,
        lm 0 0 + rm 0 0 - 2 * (lm 0 0 * rm 0 0)].get k) := by
    intro k
    fin_cases k <;> simp [ms, circuitMessageCandidates, matrixSub, matrixAdd,
      matrixMulScalarLeft, matrixPolynomial, honeMessage, List.get, Matrix.mul_apply]
    ring
  have hpublic : ∀ k, ps k =
      ([0, onePublic, lp, onePublic - lp, lp * digits,
        lp + rp - (2 : ExactPoly q n) • (lp * digits)].get k) := by
    intro k
    fin_cases k <;> simp [ps, circuitPublicCandidates, matrixSub, matrixAdd, matrixMul, List.get]
    funext row column
    simp [matrixMulScalarRight, matrixPolynomial, Matrix.smul_apply, mul_comm]
  have hcandidates : ∀ k, BooleanEncodingWithin secret (ps k) (ms k 0 0) (cs k)
      (factor * B) := by
    intro k
    rw [hmessage, hpublic]
    exact candidates k
  have hselected := generated_selected_encoding_within
    (factor * B) kind secret cs ps ms sc sp sm hcandidates hc hp hm
  have hbounded := generated_masked_encoding_within lane
    (factor * B) active (if decide ((lane.val : Int) ≤ active - 1) then 1 else 0)
    secret sc sp (outCipher lane) (outPublic lane) sm (outMessage lane) hselected rfl
    (by simpa [matrixSub] using hmc) (by simpa [matrixSub] using hmp)
    (by simpa [matrixSub] using hmm)
  refine ⟨hbounded, ?_⟩
  change outMessage lane 0 0 = 0 ∨ outMessage lane 0 0 = 1
  obtain ⟨position, hposition, hsm⟩ := hm
  have hpositionKind : position = kind := Fin.ext (by omega)
  subst position
  obtain ⟨mask, _, hmask⟩ := hmm
  fin_cases mask
  · exact Or.inl (by simpa [List.get, matrixSub] using congrArg (fun m => m 0 0) hmask)
  · have hms : sm 0 0 = ms kind 0 0 := congrArg (fun m => m 0 0) hsm
    rw [show outMessage lane = sm from hmask, hms, hmessage]
    exact boolean_gate_message_closed _ _ hlBool hrBool kind

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
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_33
        DiamondBackend.backend params layer
        (current.1, current.2.1, current.2.2.1, activeCounts, oneCipher, kinds, leftSources, rightSources,
          onePublic, oneMessage, ()) next) count initial output) :
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

#print axioms generated_circuit_lane_facts
#print axioms generated_circuit_layer_within
#print axioms generated_circuit_iteration_within

end DiamondGeneratedProof
