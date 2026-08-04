import MxxWe.DiamondFamilyChecker

open MxxWe.Generated.DiamondWeFamily
open MxxWe.Proofs.DiamondWeFamily

private def parseNat (value : String) : IO Nat :=
  match value.toNat? with
  | some result => pure result
  | none => throw <| IO.userError s!"invalid natural number: {value}"

private def parseInt (value : String) : IO Int :=
  match value.toInt? with
  | some result => pure result
  | none => throw <| IO.userError s!"invalid integer: {value}"

private def parseRat (numerator denominator : String) : IO Rat := do
  let numerator ← parseInt numerator
  let denominator ← parseNat denominator
  if denominator = 0 then
    throw <| IO.userError "a rational denominator must be positive"
  pure ((numerator : Rat) / (denominator : Rat))

def main (args : List String) : IO UInt32 := do
  if args.length != 17 then
    IO.eprintln "expected 13 integer arguments and two exact numerator/denominator pairs"
    return 2
  match args with
  | [instanceWidth, witnessWidth, depth, maxLayerWidth, ringDimension, inputCount, digitBase,
      batchBits, digitCount, modulus, gadgetBase, errorBound, preimageBound,
      trapdoorNumerator, trapdoorDenominator, errorNumerator, errorDenominator] =>
      let p : DiamondWeFamilyParams :=
        { instanceWidth := ← parseNat instanceWidth
          witnessWidth := ← parseNat witnessWidth
          depth := ← parseNat depth
          maxLayerWidth := ← parseNat maxLayerWidth
          diamondRingDimension := ← parseNat ringDimension
          diamondInputCount := ← parseNat inputCount
          diamondDigitBase := ← parseNat digitBase
          diamondBatchBits := ← parseNat batchBits
          diamondDigitCount := ← parseNat digitCount
          diamondModulus := ← parseInt modulus
          diamondGadgetBase := ← parseInt gadgetBase
          diamondErrorMaxCoefficientBound := ← parseInt errorBound
          diamondPreimageMaxCoefficientBound := ← parseInt preimageBound
          diamondTrapdoorSigma := ← parseRat trapdoorNumerator trapdoorDenominator
          diamondErrorSigma := ← parseRat errorNumerator errorDenominator }
      IO.println <| if diamondWeFamilyChecker p then "true" else "false"
      return 0
  | _ =>
      IO.eprintln "internal argument-count mismatch"
      return 2
