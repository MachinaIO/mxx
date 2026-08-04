import MxxWe.GenericInputInjection

open Mxx

namespace MxxWe

noncomputable section

/-! Parameter-generic input-injection algebra used by the Diamond execution proof.  References to
generated executable nodes are supplied and validated by the certificate layer; this module does
not identify semantic operations by generated node number. -/

/-- One concrete input-injection transition, stated independently of a fixed ring dimension or
matrix width.  These are exactly the facts extracted from the generated transition/preimage
nodes before applying the sequential-loop invariant. -/
structure InputInjectionStep (q ringDimension stateColumns : Nat) where
  state : Mxx.Matrix
  signal : Mxx.Matrix
  base : Mxx.Matrix
  stateError : Mxx.Matrix
  transition : Mxx.Matrix
  selector : Mxx.Matrix
  nextBase : Mxx.Matrix
  transitionError : Mxx.Matrix
  stateShape : Mxx.Toolkit.MatrixShape state q ringDimension 1 stateColumns
  signalShape : Mxx.Toolkit.MatrixShape signal q ringDimension 1 2
  baseShape : Mxx.Toolkit.MatrixShape base q ringDimension 2 stateColumns
  stateErrorShape : Mxx.Toolkit.MatrixShape stateError q ringDimension 1 stateColumns
  transitionShape :
    Mxx.Toolkit.MatrixShape transition q ringDimension stateColumns stateColumns
  selectorShape : Mxx.Toolkit.MatrixShape selector q ringDimension 2 2
  nextBaseShape : Mxx.Toolkit.MatrixShape nextBase q ringDimension 2 stateColumns
  transitionErrorShape :
    Mxx.Toolkit.MatrixShape transitionError q ringDimension 2 stateColumns
  stateEquation :
    Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns state =
      Mxx.Toolkit.matrixValue q ringDimension 1 2 signal *
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base +
        Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns stateError
  transitionEquation :
    Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base *
        Mxx.Toolkit.matrixValue q ringDimension stateColumns stateColumns transition =
      Mxx.Toolkit.matrixValue q ringDimension 2 2 selector *
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns nextBase +
        Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns transitionError

/-- The executable matrix operations implement the exact symbolic input-injection rewrite. -/
theorem InputInjectionStep.value_equation
    {q ringDimension stateColumns : Nat} [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    (step : InputInjectionStep q ringDimension stateColumns) :
    Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns
        (Mxx.matrixMul step.state step.transition) =
      (Mxx.Toolkit.matrixValue q ringDimension 1 2 step.signal *
          Mxx.Toolkit.matrixValue q ringDimension 2 2 step.selector) *
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns step.nextBase +
        Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns
          (propagatedStateNoise step.signal step.transitionError step.stateError
            step.transition) := by
  have stateProduct := Mxx.Toolkit.matrixValue_mul q ringDimension 1 stateColumns stateColumns
    step.state step.transition
      ⟨step.stateShape.modulus, step.stateShape.ringDimension, step.stateShape.rows,
        step.stateShape.columns⟩
      ⟨step.transitionShape.modulus, step.transitionShape.ringDimension,
        step.transitionShape.rows, step.transitionShape.columns⟩
  have signalErrorProduct := Mxx.Toolkit.matrixValue_mul q ringDimension 1 2 stateColumns
    step.signal step.transitionError
      ⟨step.signalShape.modulus, step.signalShape.ringDimension, step.signalShape.rows,
        step.signalShape.columns⟩
      ⟨step.transitionErrorShape.modulus, step.transitionErrorShape.ringDimension,
        step.transitionErrorShape.rows, step.transitionErrorShape.columns⟩
  have oldErrorProduct := Mxx.Toolkit.matrixValue_mul q ringDimension 1 stateColumns stateColumns
    step.stateError step.transition
      ⟨step.stateErrorShape.modulus, step.stateErrorShape.ringDimension,
        step.stateErrorShape.rows, step.stateErrorShape.columns⟩
      ⟨step.transitionShape.modulus, step.transitionShape.ringDimension,
        step.transitionShape.rows, step.transitionShape.columns⟩
  have signalErrorShape :=
    Mxx.Toolkit.matrixMul_shape step.signal step.transitionError step.signalShape
      step.transitionErrorShape
  have oldErrorShape :=
    Mxx.Toolkit.matrixMul_shape step.stateError step.transition step.stateErrorShape
      step.transitionShape
  have noiseValue :
      Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns
          (propagatedStateNoise step.signal step.transitionError step.stateError
            step.transition) =
        Mxx.Toolkit.matrixValue q ringDimension 1 2 step.signal *
            Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns step.transitionError +
          Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns step.stateError *
            Mxx.Toolkit.matrixValue q ringDimension stateColumns stateColumns step.transition := by
    rw [propagatedStateNoise,
      Mxx.Toolkit.matrixValue_add q ringDimension 1 stateColumns _ _
        ⟨signalErrorShape.modulus, signalErrorShape.ringDimension, signalErrorShape.rows,
          signalErrorShape.columns⟩
        ⟨oldErrorShape.modulus, oldErrorShape.ringDimension, oldErrorShape.rows,
          oldErrorShape.columns⟩,
      signalErrorProduct, oldErrorProduct]
  rw [stateProduct, noiseValue]
  exact injection_transition
    (Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns step.state)
    (Mxx.Toolkit.matrixValue q ringDimension 1 2 step.signal)
    (Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns step.base)
    (Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns step.stateError)
    (Mxx.Toolkit.matrixValue q ringDimension stateColumns stateColumns step.transition)
    (Mxx.Toolkit.matrixValue q ringDimension 2 2 step.selector)
    (Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns step.nextBase)
    (Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns step.transitionError)
    step.stateEquation step.transitionEquation

/-- One generated transition advances the deterministic worst-case recurrence used by the
parameter checker.  No probabilistic or CLT estimate occurs here. -/
theorem InputInjectionStep.noise_norm_le
    {q ringDimension stateColumns signalBound stateErrorBound errorBound preimageBound : Nat}
    [NeZero q]
    (step : InputInjectionStep q ringDimension stateColumns)
    (signalNorm : Mxx.maxCenteredCoefficientNorm step.signal ≤ signalBound)
    (stateErrorNorm : Mxx.maxCenteredCoefficientNorm step.stateError ≤ stateErrorBound)
    (transitionErrorNorm :
      Mxx.maxCenteredCoefficientNorm step.transitionError ≤ errorBound)
    (transitionNorm : Mxx.maxCenteredCoefficientNorm step.transition ≤ preimageBound) :
    Mxx.maxCenteredCoefficientNorm
        (propagatedStateNoise step.signal step.transitionError step.stateError
          step.transition) ≤
      (injectionStep ringDimension 2 stateColumns errorBound preimageBound
        (signalBound, stateErrorBound)).2 := by
  simpa [injectionStep, Nat.add_comm] using
    propagatedStateNoise_norm_le q ringDimension 2 stateColumns signalBound errorBound
      stateErrorBound preimageBound step.signal step.transitionError step.stateError
      step.transition step.signalShape step.transitionErrorShape step.stateErrorShape
      step.transitionShape signalNorm transitionErrorNorm stateErrorNorm transitionNorm

/-- A parameter-generic execution trace for the input-injection scan.  The bounds in the indices
are the exact `injectionStep` recurrence, so induction over this relation works for any runtime
`diamond_input_count`. -/
inductive InputInjectionTrace
    (q ringDimension stateColumns errorBound preimageBound : Nat) :
    Nat × Nat → Mxx.Matrix → Mxx.Matrix → Nat × Nat → Mxx.Matrix → Mxx.Matrix → Prop
  | nil (bounds signal stateError) :
      InputInjectionTrace q ringDimension stateColumns errorBound preimageBound
        bounds signal stateError bounds signal stateError
  | cons {bounds finalBounds : Nat × Nat}
      {signal stateError finalSignal finalError : Mxx.Matrix}
      (step : InputInjectionStep q ringDimension stateColumns)
      (signal_eq : step.signal = signal)
      (stateError_eq : step.stateError = stateError)
      (signalNorm : Mxx.maxCenteredCoefficientNorm signal ≤ bounds.1)
      (stateErrorNorm : Mxx.maxCenteredCoefficientNorm stateError ≤ bounds.2)
      (selectorNorm : Mxx.maxCenteredCoefficientNorm step.selector ≤ 1)
      (transitionErrorNorm :
        Mxx.maxCenteredCoefficientNorm step.transitionError ≤ errorBound)
      (transitionNorm : Mxx.maxCenteredCoefficientNorm step.transition ≤ preimageBound)
      (tail : InputInjectionTrace q ringDimension stateColumns errorBound preimageBound
        (injectionStep ringDimension 2 stateColumns errorBound preimageBound bounds)
        (Mxx.matrixMul signal step.selector)
        (propagatedStateNoise signal step.transitionError stateError step.transition)
        finalBounds finalSignal finalError) :
      InputInjectionTrace q ringDimension stateColumns errorBound preimageBound
        bounds signal stateError finalBounds finalSignal finalError

/-- Every arbitrary-length input-injection trace satisfies the exact hard-bound recurrence. -/
theorem InputInjectionTrace.norms
    {q ringDimension stateColumns errorBound preimageBound : Nat} [NeZero q]
    {initialBounds finalBounds : Nat × Nat}
    {initialSignal initialError finalSignal finalError : Mxx.Matrix}
    (trace : InputInjectionTrace q ringDimension stateColumns errorBound preimageBound
      initialBounds initialSignal initialError finalBounds finalSignal finalError)
    (initialSignalNorm :
      Mxx.maxCenteredCoefficientNorm initialSignal ≤ initialBounds.1)
    (initialErrorNorm :
      Mxx.maxCenteredCoefficientNorm initialError ≤ initialBounds.2) :
    Mxx.maxCenteredCoefficientNorm finalSignal ≤ finalBounds.1 ∧
      Mxx.maxCenteredCoefficientNorm finalError ≤ finalBounds.2 := by
  induction trace with
  | nil => exact ⟨initialSignalNorm, initialErrorNorm⟩
  | @cons bounds _ signal stateError _ _ step signal_eq stateError_eq signalNorm stateErrorNorm
      selectorNorm
      transitionErrorNorm transitionNorm tail induction =>
      have nextSignalNorm :
          Mxx.maxCenteredCoefficientNorm (Mxx.matrixMul step.signal step.selector) ≤
            productBound ringDimension 2 bounds.1 1 := by
        simpa [productBound] using
          Mxx.Toolkit.matrixMul_norm_le q ringDimension 2 bounds.1 1 step.signal
            step.selector step.signalShape.modulus step.selectorShape.modulus
            step.signalShape.ringDimension step.selectorShape.ringDimension
            step.signalShape.columns step.selectorShape.rows (by simpa [signal_eq] using signalNorm)
            selectorNorm
      have nextErrorNorm := step.noise_norm_le (by simpa [signal_eq] using signalNorm)
        (by simpa [stateError_eq] using stateErrorNorm) transitionErrorNorm transitionNorm
      subst signal
      subst stateError
      exact induction nextSignalNorm nextErrorNorm

/-- Two stages that decompose equal right public keys with equal gadget parameters obtain the
same normalized decomposition.  This is the only bridge needed for AND/XOR; the decomposition is
recomputed and never exported as a protocol artifact. -/
theorem rhsDecompositions_equal
    (samplers : MxxSamplerFamily) (contract : MxxBoundedSamplerContract samplers)
    (params : Mxx.SamplerParams) (base : Int) (digitCount : Nat)
    (encryptPublicKey decryptPublicKey encryptDecomposition decryptDecomposition : Mxx.Matrix)
    (publicKeysEqual : encryptPublicKey = decryptPublicKey)
    (encryptMember : encryptDecomposition ∈
      samplers.gadgetDecompose params base digitCount encryptPublicKey)
    (decryptMember : decryptDecomposition ∈
      samplers.gadgetDecompose params base digitCount decryptPublicKey) :
    encryptDecomposition.withSamplerParams params =
      decryptDecomposition.withSamplerParams params := by
  subst decryptPublicKey
  exact contract.gadgetDecomposeUnique params base digitCount encryptPublicKey
    encryptDecomposition decryptDecomposition encryptMember decryptMember

end

end MxxWe
