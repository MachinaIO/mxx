import MxxWe.DiamondWE.Exact

set_option maxHeartbeats 1000000

namespace Mxx.We.DiamondWE

open Mxx.Primitives

/- The bound is a finite, program-shaped recurrence.  Its environment is obtained from the
   validated Diamond parameters; no caller supplies an independent recurrence or noise value. -/
private def parameterValue (parameters : Parameters) (circuitLayers : Nat) :
    DiamondBoundParameter → Nat
  | .modulus => parameters.modulus
  | .ringDimension => parameters.ringDimension
  | .stateRows => 2
  | .stateColumns => 2 * (parameters.gadgetDigitCount + 2)
  | .errorCoefficientBound => parameters.errorCutoff
  | .preimageCoefficientBound => parameters.preimageCutoff
  | .gadgetColumns => parameters.gadgetDigitCount
  | .gadgetDecompositionBound => parameters.gadgetBase - 1
  | .inputSteps => parameters.inputCount
  | .circuitLayers => circuitLayers

private def parameter (name : DiamondBoundParameter) :
    Mxx.Primitives.BoundExpr DiamondBoundParameter :=
  .parameter name

private def literal (value : Nat) : Mxx.Primitives.BoundExpr DiamondBoundParameter :=
  .literal value

private def add (left right : Mxx.Primitives.BoundExpr DiamondBoundParameter) :
    Mxx.Primitives.BoundExpr DiamondBoundParameter :=
  .add left right

private def product (left right : Mxx.Primitives.BoundExpr DiamondBoundParameter) :
    Mxx.Primitives.BoundExpr DiamondBoundParameter :=
  .mul left right

private def inputStep (state :
    Mxx.Primitives.BoundExpr DiamondBoundParameter ×
      Mxx.Primitives.BoundExpr DiamondBoundParameter) :
    Mxx.Primitives.BoundExpr DiamondBoundParameter ×
      Mxx.Primitives.BoundExpr DiamondBoundParameter :=
  let carrier := state.1
  let noise := state.2
  -- C_next = r*n*C, with Diamond's fixed r = 2.
  let nextCarrier := product (product (literal 2) (parameter .ringDimension)) carrier
  -- I_next = r*n*C*epsilon + c*n*I*kappa.
  let nextNoise := add
    (product (product (product (literal 2) (parameter .ringDimension)) carrier)
      (parameter .errorCoefficientBound))
    (product (product (product (parameter .stateColumns)
      (parameter .ringDimension)) noise) (parameter .preimageCoefficientBound))
  (nextCarrier, nextNoise)

private def inputFold : Nat →
    Mxx.Primitives.BoundExpr DiamondBoundParameter ×
      Mxx.Primitives.BoundExpr DiamondBoundParameter →
    Mxx.Primitives.BoundExpr DiamondBoundParameter ×
      Mxx.Primitives.BoundExpr DiamondBoundParameter
  | 0, state => state
  | steps + 1, state => inputFold steps (inputStep state)

private def circuitStep (bound : Mxx.Primitives.BoundExpr DiamondBoundParameter) :=
  -- B_next = (2*A + 4)*B, where A = g*n*delta.
  product
    (add (product (literal 2) (product (product (parameter .gadgetColumns)
      (parameter .ringDimension)) (parameter .gadgetDecompositionBound))) (literal 4)) bound

private def circuitFold : Nat →
    Mxx.Primitives.BoundExpr DiamondBoundParameter →
    Mxx.Primitives.BoundExpr DiamondBoundParameter
  | 0, bound => bound
  | layers + 1, bound => circuitFold layers (circuitStep bound)

/- `C₀ = 1`, `I₀ = ε`; each input step uses the tight `r*n` coefficient bound.  `P` is the
   payload error, `A` the decomposition error, `B` the circuit-layer error, and `F` the fuse
   error.  In particular, no product introduces the old quadratic `n²` factor. -/
def deriveOutputNoiseBound (parameters : Parameters) (circuitLayers : Nat) :
    Mxx.Primitives.BoundExpr DiamondBoundParameter :=
  let input := inputFold parameters.inputCount
    (literal 1, parameter .errorCoefficientBound)
  let payload := product (product (parameter .stateColumns) (parameter .ringDimension))
    (product input.2 (parameter .preimageCoefficientBound))
  let circuit := circuitFold circuitLayers payload
  -- F = 2*P + g*n*(P + B_l)*delta.
  add (product (literal 2) payload)
    (product (product (parameter .gadgetColumns) (parameter .ringDimension))
      (product (add payload circuit) (parameter .gadgetDecompositionBound)))

def deriveBoundEnvironment (parameters : Parameters) (circuitLayers : Nat) :
    DiamondBoundParameter → Nat :=
  parameterValue parameters circuitLayers

def inputBoundsFrom (parameters : Parameters) : Nat → Nat × Nat → Nat × Nat
  | 0, state => state
  | steps + 1, state => inputBoundsFrom parameters steps
      (2 * parameters.ringDimension * state.1,
        2 * parameters.ringDimension * state.1 * parameters.errorCutoff +
          (2 * (parameters.gadgetDigitCount + 2)) * parameters.ringDimension * state.2 *
            parameters.preimageCutoff)

def inputBounds (parameters : Parameters) : Nat → Nat × Nat := fun steps =>
  inputBoundsFrom parameters steps (1, parameters.errorCutoff)

def circuitBoundFrom (parameters : Parameters) : Nat → Nat → Nat
  | 0, payload => payload
  | layers + 1, payload => circuitBoundFrom parameters layers
        ((2 * (parameters.gadgetDigitCount * parameters.ringDimension *
          (parameters.gadgetBase - 1)) + 4) * payload)

def circuitBound (parameters : Parameters) (layers : Nat) (payload : Nat) : Nat :=
  circuitBoundFrom parameters layers payload

private def evalPair (environment : DiamondBoundParameter → Nat)
    (state : Mxx.Primitives.BoundExpr DiamondBoundParameter ×
      Mxx.Primitives.BoundExpr DiamondBoundParameter) : Nat × Nat :=
  (state.1.eval environment, state.2.eval environment)

private theorem inputFold_evaluates (parameters : Parameters) (circuitLayers steps : Nat)
    (state : Mxx.Primitives.BoundExpr DiamondBoundParameter ×
      Mxx.Primitives.BoundExpr DiamondBoundParameter) :
    evalPair (deriveBoundEnvironment parameters circuitLayers) (inputFold steps state) =
      inputBoundsFrom parameters steps (evalPair
        (deriveBoundEnvironment parameters circuitLayers) state) := by
  induction steps generalizing state with
  | zero => rfl
  | succ steps ih =>
      simp only [inputFold]
      rw [ih (inputStep state)]
      apply congrArg (inputBoundsFrom parameters steps)
      apply Prod.ext
      · simp [inputStep, evalPair, deriveBoundEnvironment, parameterValue, product, literal,
          parameter, Mxx.Primitives.BoundExpr.eval]
      · simp [inputStep, evalPair, deriveBoundEnvironment, parameterValue, product, literal,
          parameter, add, Mxx.Primitives.BoundExpr.eval]

private theorem circuitFold_evaluates (parameters : Parameters) (steps : Nat)
    (circuitLayers : Nat) (bound : Mxx.Primitives.BoundExpr DiamondBoundParameter) :
    (circuitFold steps bound).eval (deriveBoundEnvironment parameters circuitLayers) =
      circuitBoundFrom parameters steps (bound.eval
        (deriveBoundEnvironment parameters circuitLayers)) := by
  induction steps generalizing bound with
  | zero => rfl
  | succ steps ih =>
      simp only [circuitFold]
      rw [ih (circuitStep bound)]
      apply congrArg (circuitBoundFrom parameters steps)
      simp [circuitStep, deriveBoundEnvironment, parameterValue, product, literal, parameter,
        add, Mxx.Primitives.BoundExpr.eval]

theorem deriveOutputNoiseBound_evaluates (parameters : Parameters) (circuitLayers : Nat) :
    (deriveOutputNoiseBound parameters circuitLayers).eval
        (deriveBoundEnvironment parameters circuitLayers) =
      let input := inputBounds parameters parameters.inputCount
      let payload := 2 * (parameters.gadgetDigitCount + 2) * parameters.ringDimension *
        input.2 * parameters.preimageCutoff
      let circuit := circuitBound parameters circuitLayers payload
      2 * payload + parameters.gadgetDigitCount * parameters.ringDimension *
        (payload + circuit) * (parameters.gadgetBase - 1) := by
  unfold deriveOutputNoiseBound
  dsimp only
  have input := inputFold_evaluates parameters circuitLayers parameters.inputCount
    (literal 1, parameter .errorCoefficientBound)
  have circuit := circuitFold_evaluates parameters circuitLayers circuitLayers
    ((product (product (parameter .stateColumns) (parameter .ringDimension))
      (product (inputFold parameters.inputCount
        (literal 1, parameter .errorCoefficientBound)).2
        (parameter .preimageCoefficientBound))))
  have inputNoise :
      (inputFold parameters.inputCount
        (literal 1, parameter .errorCoefficientBound)).2.eval
          (deriveBoundEnvironment parameters circuitLayers) =
        (inputBounds parameters parameters.inputCount).2 :=
    congrArg Prod.snd input
  have inputNoise' := inputNoise
  change (inputFold parameters.inputCount
      (literal 1, parameter .errorCoefficientBound)).2.eval
        (parameterValue parameters circuitLayers) = _ at inputNoise'
  simp only [add, product, Mxx.Primitives.BoundExpr.eval]
  rw [inputNoise]
  have circuit' := circuit
  simp only [product] at circuit'
  simp only [Mxx.Primitives.BoundExpr.eval] at circuit'
  rw [inputNoise] at circuit'
  rw [circuit']
  simp [deriveBoundEnvironment, parameterValue, circuitBound, inputBounds]
  ring_nf

/- This certificate is the exact/noise handoff.  Its approximation is produced by Exact.lean from
   generated node equations; Noise.lean only consumes it and applies the recurrence. -/
structure OutputNoiseCertificate (candidate : Candidate) (bound : Nat) where
  noisyPlaintext : ExactMatrix candidate.parameters.modulus candidate.parameters.ringDimension 1 1
  ideal : ExactMatrix candidate.parameters.modulus candidate.parameters.ringDimension 1 1
  approximation : ApproxWithin noisyPlaintext ideal bound

theorem output_noise_norm_le {candidate : Candidate} {bound : Nat}
    (certificate : OutputNoiseCertificate candidate bound) :
    matrixNorm certificate.approximation.toApprox.error ≤ bound := by
  exact certificate.approximation.norm_le

end Mxx.We.DiamondWE
