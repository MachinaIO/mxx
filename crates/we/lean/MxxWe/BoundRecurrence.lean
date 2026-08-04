namespace MxxWe

/-! Deterministic worst-case recurrences that compose executable sampler bounds across the
Diamond input-injection, Boolean-circuit, and decoder stages.  Individual sampler support bounds
come from the IR semantics and sampler contract; this module only combines them. -/

def productBound (ringDimension innerDimension left right : Nat) : Nat :=
  ringDimension * innerDimension * left * right

def injectionStep (ringDimension stateRows stateColumns error preimage : Nat)
    (state : Nat × Nat) : Nat × Nat :=
  let secret := productBound ringDimension stateRows state.1 1
  let propagated := productBound ringDimension stateColumns state.2 preimage
  let injected := productBound ringDimension stateRows state.1 error
  (secret, propagated + injected)

def injectionBound (ringDimension stateRows stateColumns inputCount error preimage : Nat) : Nat :=
  let step := injectionStep ringDimension stateRows stateColumns error preimage
  (List.range inputCount).foldl (fun state _ => step state) (1, error) |>.2

def gateStep (ringDimension publicColumns digitBound oneError input : Nat) : Nat :=
  let zeroBound := 2 * oneError
  let notBound := oneError + input
  let product := productBound ringDimension publicColumns input digitBound +
    productBound ringDimension 1 input 1
  let xorBound := 2 * input + 2 * product
  max zeroBound (max oneError (max input (max notBound (max product xorBound))))

def circuitBound (ringDimension publicColumns layerCount digitBound oneError : Nat) : Nat :=
  (List.range layerCount).foldl
    (fun input _ => gateStep ringDimension publicColumns digitBound oneError input)
    (2 * oneError)

def diamondFinalBound (ringDimension stateRows stateColumns publicColumns inputCount
    layerCount gadgetBase error preimage : Nat) : Nat :=
  let stateError :=
    injectionBound ringDimension stateRows stateColumns inputCount error preimage
  let oneError := productBound ringDimension stateColumns stateError preimage
  let digitBound := max (gadgetBase / 2) 1
  let output := circuitBound ringDimension publicColumns layerCount digitBound oneError
  let difference := oneError + output
  let projected := productBound ringDimension publicColumns difference digitBound
  oneError + oneError + projected

def diamondParamsValid (ringDimension stateRows stateColumns publicColumns inputCount layerCount
    gadgetBase modulus : Nat) : Bool :=
  0 < ringDimension &&
  stateRows = 2 &&
  0 < publicColumns &&
  stateColumns = stateRows * (publicColumns + 2) &&
  0 < inputCount &&
  0 < layerCount &&
  1 < gadgetBase &&
  4 ≤ modulus

def diamondChecker (ringDimension stateRows stateColumns publicColumns inputCount layerCount
    gadgetBase error preimage modulus : Nat) : Bool :=
  diamondParamsValid ringDimension stateRows stateColumns publicColumns inputCount layerCount
    gadgetBase modulus &&
  diamondFinalBound ringDimension stateRows stateColumns publicColumns inputCount layerCount
      gadgetBase error preimage < modulus / 4

theorem diamondChecker_eq_true_iff (ringDimension stateRows stateColumns publicColumns
    inputCount layerCount gadgetBase error preimage modulus : Nat) :
    diamondChecker ringDimension stateRows stateColumns publicColumns inputCount layerCount
      gadgetBase error preimage modulus = true ↔
    diamondParamsValid ringDimension stateRows stateColumns publicColumns inputCount layerCount
        gadgetBase modulus = true ∧
      diamondFinalBound ringDimension stateRows stateColumns publicColumns inputCount layerCount
        gadgetBase error preimage < modulus / 4 := by
  simp [diamondChecker]

theorem diamondChecker_sound (ringDimension stateRows stateColumns publicColumns inputCount
    layerCount gadgetBase error preimage modulus actualError : Nat)
    (accepted : diamondChecker ringDimension stateRows stateColumns publicColumns inputCount
      layerCount gadgetBase error preimage modulus = true)
    (dominated : actualError ≤ diamondFinalBound ringDimension stateRows stateColumns
      publicColumns inputCount layerCount gadgetBase error preimage) :
    actualError < modulus / 4 := by
  exact Nat.lt_of_le_of_lt dominated
    ((diamondChecker_eq_true_iff _ _ _ _ _ _ _ _ _ _).mp accepted).2

theorem diamondChecker_params_valid (ringDimension stateRows stateColumns publicColumns
    inputCount layerCount gadgetBase error preimage modulus : Nat)
    (accepted : diamondChecker ringDimension stateRows stateColumns publicColumns inputCount
      layerCount gadgetBase error preimage modulus = true) :
    diamondParamsValid ringDimension stateRows stateColumns publicColumns inputCount layerCount
      gadgetBase modulus = true :=
  ((diamondChecker_eq_true_iff _ _ _ _ _ _ _ _ _ _).mp accepted).1

end MxxWe
