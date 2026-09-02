import MxxPrimitives.Bounds

namespace Mxx.We.DiamondWE

/- The application parameter record deliberately contains only concrete values.  A generated
   candidate carries the proof that these values satisfy the protocol-side invariants. -/
structure ParametersData where
  modulus : Nat
  ringDimension : Nat
  inputCount : Nat
  digitBase : Nat
  batchBits : Nat
  gadgetBase : Nat
  gadgetDigitCount : Nat
  witnessWidth : Nat
  errorCutoff : Nat
  preimageCutoff : Nat
  deriving Repr, DecidableEq

def decoderQuarter (q : Nat) : Nat := (q - 2 + 2) / 4

structure DecoderGeometryValid (q : Nat) : Prop where
  quarter_le_half : decoderQuarter q ≤ q / 2
  half_le_three_quarters : q / 2 ≤ 3 * decoderQuarter q
  three_quarters_le_q : 3 * decoderQuarter q ≤ q

def ParametersData.Valid (data : ParametersData) : Prop :=
  1 < data.modulus ∧
    0 < data.ringDimension ∧
    0 < data.inputCount ∧
    0 < data.digitBase ∧
    0 < data.batchBits ∧
    1 < data.gadgetBase ∧
    0 < data.gadgetDigitCount ∧
    0 < data.witnessWidth ∧
    0 < data.errorCutoff ∧
    0 < data.preimageCutoff ∧
    DecoderGeometryValid data.modulus

theorem ParametersData.geometryValid {data : ParametersData} (h : data.Valid) :
    DecoderGeometryValid data.modulus := by
  rcases h with ⟨_, _, _, _, _, _, _, _, _, _, geometry⟩
  exact geometry

structure Parameters where
  data : ParametersData
  valid : data.Valid

namespace Parameters

abbrev modulus (parameters : Parameters) : Nat := parameters.data.modulus
abbrev ringDimension (parameters : Parameters) : Nat := parameters.data.ringDimension
abbrev inputCount (parameters : Parameters) : Nat := parameters.data.inputCount
abbrev digitBase (parameters : Parameters) : Nat := parameters.data.digitBase
abbrev batchBits (parameters : Parameters) : Nat := parameters.data.batchBits
abbrev gadgetBase (parameters : Parameters) : Nat := parameters.data.gadgetBase
abbrev gadgetDigitCount (parameters : Parameters) : Nat := parameters.data.gadgetDigitCount
abbrev witnessWidth (parameters : Parameters) : Nat := parameters.data.witnessWidth
abbrev errorCutoff (parameters : Parameters) : Nat := parameters.data.errorCutoff
abbrev preimageCutoff (parameters : Parameters) : Nat := parameters.data.preimageCutoff

end Parameters

inductive DiamondBoundParameter
  | modulus
  | ringDimension
  | stateRows
  | stateColumns
  | gadgetColumns
  | errorCoefficientBound
  | preimageCoefficientBound
  | gadgetDecompositionBound
  | inputSteps
  | circuitLayers
  deriving Repr, DecidableEq

structure BoundData where
  expression : Mxx.Primitives.BoundExpr DiamondBoundParameter
  environment : DiamondBoundParameter → Nat
  value : Nat
  evaluated : expression.eval environment = value

example : ParametersData.Valid
    { modulus := 17, ringDimension := 4,
      inputCount := 1, digitBase := 4, batchBits := 2,
      gadgetBase := 2, gadgetDigitCount := 3, witnessWidth := 1, errorCutoff := 1,
      preimageCutoff := 1 } := by
  norm_num [ParametersData.Valid]
  constructor <;> norm_num [decoderQuarter]

end Mxx.We.DiamondWE

namespace Mxx.We.Golden.DiamondWE

/- Generated fixtures provide this definition.  The reusable library intentionally does not
   invent a checked-in candidate while the linked IR and runtime backend are still evolving. -/
-- def parameters : Mxx.We.DiamondWE.Parameters := ...

end Mxx.We.Golden.DiamondWE
