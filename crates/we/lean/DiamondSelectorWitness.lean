import DiamondProofParameters
import DiamondAccumulatedSecretProof
import DiamondIndexProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

/-- A view of the witnesses of the generated preprocessing scope. The coordinate
    equations and scan refer to the original slot, not an independent selector. -/
structure InjectorSelectorWitness
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner) where
  regular : ExactMatrix q n 2 2
  zeroState : ExactMatrix q n 2 2
  initial : ExactMatrix q n 2 2
  selector : ExactMatrix q n 2 2
  error : ExactMatrix q n 2 inner
  state : Int
  digit : Int
  firstNew : Int
  stateEquation : state = (slot : Int) %
    (1 + params.diamond_batch_bits * params.diamond_input_count)
  digitEquation : digit = ((slot : Int) /
    (1 + params.diamond_batch_bits * params.diamond_input_count)) % params.diamond_digit_base
  firstNewEquation : firstNew = ((slot : Int) /
    (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
      params.diamond_digit_base)) * params.diamond_batch_bits + 1
  regularRun : concatDiagonal secret secret regular
  zeroRun : concatDiagonal secret (1 : ExactMatrix q n 1 1) zeroState
  initialRun : select (if decide (state = 0) then 1 else 0) [regular, zeroState] initial
  scanRun : MxxIR.IterRuns
    (fun bit current next ↦ Stage_encrypt.sequential_parallel_generatedRoot_72_21
      backend hashModel params slot bit (current, digit, state, firstNew, secret, ()) next)
    params.diamond_batch_bits.toNat initial selector
  errorRun : gaussianSample params.diamond_error_sigma
    params.diamond_error_max_coefficient_bound error
  targetEquation : target = selector * publicMatrix + error

theorem generated_selector_witness
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (hrun : Stage_encrypt.parallel_generatedRoot_72 backend hashModel params slot
      (secret, publicMatrix, ()) target) :
    Nonempty (InjectorSelectorWitness backend hashModel params slot secret publicMatrix target) := by
  rcases hrun with ⟨regular, initialZero, initial, selector, error,
    _, hregular, hzero, _, _, _, hselect, _, _, _, _, hscan, herror, htarget⟩
  exact ⟨{
    regular := regular
    zeroState := initialZero
    initial := initial
    selector := selector
    error := error
    state := _
    digit := _
    firstNew := _
    stateEquation := rfl
    digitEquation := rfl
    firstNewEquation := rfl
    regularRun := hregular
    zeroRun := hzero
    initialRun := hselect
    scanRun := hscan
    errorRun := herror
    targetEquation := htarget }⟩

theorem selector_witness_existing_action
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (witness : InjectorSelectorWitness backend hashModel params slot secret publicMatrix target)
    (row : ExactMatrix q n 1 2) (hstate : witness.state < witness.firstNew) :
    (row * witness.selector) 0 0 = row 0 0 * secret 0 0 ∧
      (row * witness.selector) 0 1 =
        if witness.state = 0 then row 0 1 else row 0 1 * secret 0 0 := by
  have hselect := witness.initialRun
  by_cases hzero : witness.state = 0
  · simp only [hzero, decide_true, ite_true] at hselect
    rcases hselect with ⟨position, hposition, hvalue⟩
    have hp : position = (1 : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    have hdiag : concatDiagonal secret (1 : ExactMatrix q n 1 1) witness.initial :=
      hvalue.symm ▸ witness.zeroRun
    have h := generated_existing_selector_action backend hashModel params slot _ _ _ _
      witness.initial witness.selector secret 1 row hstate hdiag witness.scanRun
    simpa [hzero] using h
  · simp only [hzero, decide_false] at hselect
    rcases hselect with ⟨position, hposition, hvalue⟩
    have hp : position = (0 : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    have hdiag : concatDiagonal secret secret witness.initial :=
      hvalue.symm ▸ witness.regularRun
    have h := generated_existing_selector_action backend hashModel params slot _ _ _ _
      witness.initial witness.selector secret secret row hstate hdiag witness.scanRun
    simpa [hzero] using h

theorem selector_witness_new_action
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (witness : InjectorSelectorWitness backend hashModel params slot secret publicMatrix target)
    (row : ExactMatrix q n 1 2) (bit : Nat)
    (hstate : witness.state = witness.firstNew + (bit : Int))
    (hbit : bit < params.diamond_batch_bits.toNat) :
    ∃ bitValue : ExactMatrix q n 1 1,
      select (if decide ((witness.digit / (2 ^ bit)) % 2 = 1) then 1 else 0)
        [0, 1] bitValue ∧
      (row * witness.selector) 0 0 = row 0 0 * secret 0 0 ∧
      (row * witness.selector) 0 1 = (row 0 0 * secret 0 0) * bitValue 0 0 := by
  exact generated_new_selector_action backend hashModel params slot _ bit _ _ _
    witness.initial witness.selector secret row hstate hbit witness.scanRun

theorem selector_witness_integer_existing_coordinate
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (witness : InjectorSelectorWitness backend hashModel params slot secret publicMatrix target)
    (row nextRow : ErrorMatrix n 1 2)
    (hrow : reduceMatrix q n 1 2 nextRow =
      reduceMatrix q n 1 2 row * witness.selector)
    (hstate : witness.state < witness.firstNew) :
    reducePoly q n (nextRow 0 1) =
      if witness.state = 0 then reducePoly q n (row 0 1)
      else reducePoly q n (row 0 1) * secret 0 0 := by
  have h := selector_witness_existing_action backend hashModel params slot secret publicMatrix
    target witness (reduceMatrix q n 1 2 row) hstate
  exact (congrArg (fun value : ExactMatrix q n 1 2 ↦ value 0 1) hrow).trans h.2

theorem selector_witness_integer_new_coordinate
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (witness : InjectorSelectorWitness backend hashModel params slot secret publicMatrix target)
    (row nextRow : ErrorMatrix n 1 2) (bit : Nat)
    (hrow : reduceMatrix q n 1 2 nextRow =
      reduceMatrix q n 1 2 row * witness.selector)
    (hstate : witness.state = witness.firstNew + (bit : Int))
    (hbit : bit < params.diamond_batch_bits.toNat) :
    ∃ bitValue : ExactMatrix q n 1 1,
      select (if decide ((witness.digit / (2 ^ bit)) % 2 = 1) then 1 else 0)
        [0, 1] bitValue ∧
      reducePoly q n (nextRow 0 1) =
        (reducePoly q n (row 0 0) * secret 0 0) * bitValue 0 0 := by
  obtain ⟨bitValue, hselect, _, hcoord⟩ := selector_witness_new_action backend hashModel params
    slot secret publicMatrix target witness (reduceMatrix q n 1 2 row) bit hstate hbit
  exact ⟨bitValue, hselect,
    (congrArg (fun value : ExactMatrix q n 1 2 ↦ value 0 1) hrow).trans hcoord⟩

theorem selector_witness_coordinates
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (witness : InjectorSelectorWitness backend hashModel params slot secret publicMatrix target)
    (stateCount digitBase layer : Nat) (hstates : 0 < stateCount) (hdigits : 0 < digitBase)
    (hstateGeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (hbaseGeometry : (digitBase : Int) = params.diamond_digit_base)
    (digit : Fin digitBase) (state : Fin stateCount)
    (hslot : slot = (layer * digitBase + digit.val) * stateCount + state.val) :
    witness.state = (state.val : Int) ∧ witness.digit = (digit.val : Int) ∧
      witness.firstNew = (layer : Int) * params.diamond_batch_bits + 1 := by
  have hwidth : ((digitBase * stateCount : Nat) : Int) =
      params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base := by
    rw [Nat.cast_mul, hstateGeometry, hbaseGeometry]
    ring
  obtain ⟨hlayer, hstate, hdigit⟩ := transition_coordinates stateCount digitBase layer
    hstates hdigits digit state
  refine ⟨?_, ?_, ?_⟩
  · rw [witness.stateEquation, hslot, ← hstateGeometry, ← Int.natCast_emod, hstate]
  · rw [witness.digitEquation, hslot, ← hstateGeometry, ← hbaseGeometry,
      ← Int.natCast_ediv, hdigit, ← Int.natCast_emod]
    congr 1
    simp [Nat.add_mod, Nat.mod_eq_of_lt digit.isLt]
  · rw [witness.firstNewEquation, hslot, ← hwidth, ← Int.natCast_ediv, hlayer]

#print axioms selector_witness_coordinates
#print axioms selector_witness_integer_existing_coordinate
#print axioms selector_witness_integer_new_coordinate
#print axioms selector_witness_new_action
#print axioms generated_selector_witness
#print axioms selector_witness_existing_action

end DiamondGeneratedProof
