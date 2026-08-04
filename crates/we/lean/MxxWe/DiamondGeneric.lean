import MxxWe.Correctness

namespace MxxWe

/-- The complete symbolic parameter interface of the fixed-maximum-shape Diamond WE graph.

The active gate counts, gate kinds, predecessor indices, and output index are protocol inputs,
not parameters.  The four Boolean fields below fix only the rectangular allocation bounds. -/
structure DiamondWeParameters where
  instanceWidth : Nat
  witnessWidth : Nat
  depth : Nat
  maxLayerWidth : Nat
  ringDimension : Nat
  inputCount : Nat
  digitBase : Nat
  batchBits : Nat
  digitCount : Nat
  modulus : Nat
  gadgetBase : Nat
  errorMaxCoefficientBound : Nat
  preimageMaxCoefficientBound : Nat
  trapdoorSigma : Rat
  errorSigma : Rat
  deriving DecidableEq, Repr

def DiamondWeParameters.inputWidth (p : DiamondWeParameters) : Nat :=
  p.instanceWidth + p.witnessWidth

def DiamondWeParameters.stateRows (_p : DiamondWeParameters) : Nat := 2

def DiamondWeParameters.stateColumns (p : DiamondWeParameters) : Nat :=
  p.stateRows * (p.digitCount + 2)

def DiamondWeParameters.publicColumns (p : DiamondWeParameters) : Nat :=
  p.digitCount

def DiamondWeParameters.digitBound (p : DiamondWeParameters) : Nat :=
  max (p.gadgetBase / 2) 1

def DiamondWeParameters.inputStateBound (p : DiamondWeParameters) : Nat :=
  injectionBound p.ringDimension p.stateRows p.stateColumns p.inputCount
    p.errorMaxCoefficientBound p.preimageMaxCoefficientBound

def DiamondWeParameters.oneEncodingBound (p : DiamondWeParameters) : Nat :=
  productBound p.ringDimension p.stateColumns p.inputStateBound
    p.preimageMaxCoefficientBound

def DiamondWeParameters.circuitNoiseBound (p : DiamondWeParameters) : Nat :=
  circuitBound p.ringDimension p.publicColumns p.depth p.digitBound p.oneEncodingBound

def DiamondWeParameters.finalBound (p : DiamondWeParameters) : Nat :=
  p.oneEncodingBound + p.oneEncodingBound +
    productBound p.ringDimension p.publicColumns
      (p.oneEncodingBound + p.circuitNoiseBound) p.digitBound

@[simp] theorem injectionBound_zero (ringDimension stateRows stateColumns error preimage : Nat) :
    injectionBound ringDimension stateRows stateColumns 0 error preimage = error := by
  simp [injectionBound]

@[simp] theorem injectionBound_succ (ringDimension stateRows stateColumns inputCount error
    preimage : Nat) :
    injectionBound ringDimension stateRows stateColumns (inputCount + 1) error preimage =
      (injectionStep ringDimension stateRows stateColumns error preimage
        ((List.range inputCount).foldl
          (fun state _ => injectionStep ringDimension stateRows stateColumns error preimage state)
          (1, error))).2 := by
  simp [injectionBound, List.range_succ, List.foldl_append]

@[simp] theorem circuitBound_zero
    (ringDimension publicColumns digitBound oneError : Nat) :
    circuitBound ringDimension publicColumns 0 digitBound oneError = 2 * oneError := by
  simp [circuitBound]

@[simp] theorem circuitBound_succ
    (ringDimension publicColumns layerCount digitBound oneError : Nat) :
    circuitBound ringDimension publicColumns (layerCount + 1) digitBound oneError =
      gateStep ringDimension publicColumns digitBound oneError
        (circuitBound ringDimension publicColumns layerCount digitBound oneError) := by
  simp [circuitBound, List.range_succ, List.foldl_append]

/-- Parameter relations required by the symbolic Diamond graph and its hard-bound proof.

Sampler sigmas remain explicit positive parameters, while the executable sampler nodes carry the
integer hard-support bounds used by the proof.  Correctness therefore does not use a probabilistic
Gaussian tail or a central-limit approximation. -/
def diamondWeParametersValid (p : DiamondWeParameters) : Bool :=
  decide (0 < p.inputWidth) &&
  decide (0 < p.depth) &&
  decide (p.inputWidth ≤ p.maxLayerWidth) &&
  decide (p.witnessWidth = p.inputCount * p.batchBits) &&
  decide (0 < p.ringDimension) &&
  decide (0 < p.inputCount) &&
  decide (0 < p.batchBits) &&
  decide (2 ^ p.batchBits ≤ p.digitBase) &&
  decide (0 < p.digitCount) &&
  decide (4 ≤ p.modulus) &&
  decide (2 ≤ p.gadgetBase) &&
  decide ((0 : Rat) < p.trapdoorSigma) &&
  decide ((0 : Rat) ≤ p.errorSigma)

/-- The parameter-only checker used by search.  It accepts exactly when every structural
parameter relation holds and the deterministic worst-case error is strictly below `q / 4`. -/
def diamondWeChecker (p : DiamondWeParameters) : Bool :=
  diamondWeParametersValid p && decide (p.finalBound < p.modulus / 4)

theorem diamondWeChecker_eq_true_iff (p : DiamondWeParameters) :
    diamondWeChecker p = true ↔
      diamondWeParametersValid p = true ∧ p.finalBound < p.modulus / 4 := by
  simp [diamondWeChecker]

theorem diamondWeChecker_parametersValid (p : DiamondWeParameters)
    (accepted : diamondWeChecker p = true) : diamondWeParametersValid p = true :=
  (diamondWeChecker_eq_true_iff p).mp accepted |>.1

theorem diamondWeChecker_finalBound_lt (p : DiamondWeParameters)
    (accepted : diamondWeChecker p = true) : p.finalBound < p.modulus / 4 :=
  (diamondWeChecker_eq_true_iff p).mp accepted |>.2

theorem diamondWeChecker_modulus_ge (p : DiamondWeParameters)
    (accepted : diamondWeChecker p = true) : 4 ≤ p.modulus := by
  have valid := diamondWeChecker_parametersValid p accepted
  simp only [diamondWeParametersValid, Bool.and_eq_true, decide_eq_true_eq] at valid
  aesop

/-- Soundness of the symbolic checker for any concrete final error dominated by the exact
worst-case recurrence. -/
theorem diamondWeChecker_sound (p : DiamondWeParameters) (actualError : Nat)
    (accepted : diamondWeChecker p = true) (dominated : actualError ≤ p.finalBound) :
    actualError < p.modulus / 4 :=
  Nat.lt_of_le_of_lt dominated (diamondWeChecker_finalBound_lt p accepted)

theorem diamondWeParameters_finalBound_eq (p : DiamondWeParameters) :
    p.finalBound =
      diamondFinalBound p.ringDimension p.stateRows p.stateColumns p.publicColumns p.inputCount
        p.depth p.gadgetBase p.errorMaxCoefficientBound p.preimageMaxCoefficientBound := by
  rfl

/-- The 15-parameter checker refines the smaller arithmetic checker used by the command-line
parameter search. -/
theorem diamondWeChecker_coreAccepted (p : DiamondWeParameters)
    (accepted : diamondWeChecker p = true) :
    diamondChecker p.ringDimension p.stateRows p.stateColumns p.publicColumns p.inputCount
      p.depth p.gadgetBase p.errorMaxCoefficientBound p.preimageMaxCoefficientBound p.modulus =
      true := by
  apply (diamondChecker_eq_true_iff _ _ _ _ _ _ _ _ _ _).2
  constructor
  · have valid := diamondWeChecker_parametersValid p accepted
    simp only [diamondWeParametersValid, Bool.and_eq_true, decide_eq_true_eq] at valid
    simp only [diamondParamsValid, DiamondWeParameters.stateRows,
      DiamondWeParameters.stateColumns, DiamondWeParameters.publicColumns, Bool.and_eq_true,
      decide_eq_true_eq]
    aesop
  · rw [← diamondWeParameters_finalBound_eq p]
    exact diamondWeChecker_finalBound_lt p accepted

/-- The generic final matrix-error bound.  Injection and circuit elaboration only need to establish
the two displayed intermediate bounds; this theorem is independent of concrete dimensions and
the dynamically supplied gate opcodes and predecessor indices. -/
theorem finalDecoderNoise_norm_le_finalBound (p : DiamondWeParameters) [NeZero p.modulus]
    (decoderError kError oneError circuitError rDecomposed : Mxx.Matrix)
    (decoderShape : Mxx.Toolkit.MatrixShape decoderError p.modulus p.ringDimension 1 1)
    (kShape : Mxx.Toolkit.MatrixShape kError p.modulus p.ringDimension 1 1)
    (oneShape :
      Mxx.Toolkit.MatrixShape oneError p.modulus p.ringDimension 1 p.publicColumns)
    (circuitShape :
      Mxx.Toolkit.MatrixShape circuitError p.modulus p.ringDimension 1 p.publicColumns)
    (rShape :
      Mxx.Toolkit.MatrixShape rDecomposed p.modulus p.ringDimension p.publicColumns 1)
    (decoderNorm : Mxx.maxCenteredCoefficientNorm decoderError ≤ p.oneEncodingBound)
    (kNorm : Mxx.maxCenteredCoefficientNorm kError ≤ p.oneEncodingBound)
    (oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ p.oneEncodingBound)
    (circuitNorm : Mxx.maxCenteredCoefficientNorm circuitError ≤ p.circuitNoiseBound)
    (rNorm : Mxx.maxCenteredCoefficientNorm rDecomposed ≤ p.digitBound) :
    Mxx.maxCenteredCoefficientNorm
        (finalDecoderNoise decoderError kError oneError circuitError rDecomposed) ≤
      p.finalBound := by
  exact finalDecoderNoise_norm_le p.modulus p.ringDimension p.publicColumns
    p.oneEncodingBound p.circuitNoiseBound p.digitBound decoderError kError oneError
    circuitError rDecomposed decoderShape kShape oneShape circuitShape rShape decoderNorm kNorm
    oneNorm circuitNorm rNorm

/-- Parameter-independent decoding theorem used by the generated protocol proof. -/
theorem diamondWeGenericDecoderCorrect (p : DiamondWeParameters) (message : Bool)
    (accepted : diamondWeChecker p = true) (actualError : Nat)
    (actual_le : actualError ≤ p.finalBound) (error : Int)
    (error_le : error.natAbs ≤ actualError) :
    decodeBooleanInterval p.modulus
      (((if message then (p.modulus : Int) / 2 else 0) + error) % p.modulus) = message := by
  apply decodeBooleanInterval_correct message p.modulus error
  · exact_mod_cast diamondWeChecker_modulus_ge p accepted
  · exact_mod_cast Nat.lt_of_le_of_lt error_le
      (diamondWeChecker_sound p actualError accepted actual_le)

/-- Recomputing a gadget decomposition from the same public matrix and the same gadget
parameters produces the same normalized matrix.  The decomposition is not ciphertext data and
does not need an artifact identity of its own. -/
theorem samePublicKeyDecomposition
    (samplers : Mxx.MxxSamplerFamily) (contract : Mxx.MxxBoundedSamplerContract samplers)
    (params : Mxx.SamplerParams) (base : Int) (digitCount : Nat)
    (publicKey left right : Mxx.Matrix)
    (leftMember : left ∈ samplers.gadgetDecompose params base digitCount publicKey)
    (rightMember : right ∈ samplers.gadgetDecompose params base digitCount publicKey) :
    left.withSamplerParams params = right.withSamplerParams params :=
  contract.gadgetDecomposeUnique params base digitCount publicKey left right leftMember rightMember

end MxxWe
