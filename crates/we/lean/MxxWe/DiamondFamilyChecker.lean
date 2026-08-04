import MxxWe.DiamondGeneric
import MxxWe.Generated.DiamondWeFamily.Statement

open MxxWe.Generated.DiamondWeFamily

namespace MxxWe.Proofs.DiamondWeFamily

/-- Converts generated family parameters to the protocol-independent hard-bound model. -/
def genericParameters (p : DiamondWeFamilyParams) : DiamondWeParameters where
  instanceWidth := p.instanceWidth
  witnessWidth := p.witnessWidth
  depth := p.depth
  maxLayerWidth := p.maxLayerWidth
  ringDimension := p.diamondRingDimension
  inputCount := p.diamondInputCount
  digitBase := p.diamondDigitBase
  batchBits := p.diamondBatchBits
  digitCount := p.diamondDigitCount
  modulus := p.diamondModulus.toNat
  gadgetBase := p.diamondGadgetBase.toNat
  errorMaxCoefficientBound := p.diamondErrorMaxCoefficientBound.toNat
  preimageMaxCoefficientBound := p.diamondPreimageMaxCoefficientBound.toNat
  trapdoorSigma := p.diamondTrapdoorSigma
  errorSigma := p.diamondErrorSigma

/--
Checks generated parameter validity, protocol shape relations, and the exact worst-case
quarter-modulus bound. This executable checker does not by itself prove end-to-end correctness.
-/
def diamondWeFamilyChecker (p : DiamondWeFamilyParams) : Bool :=
  DiamondWeFamilyParamsValid p && diamondWeChecker (genericParameters p)

end MxxWe.Proofs.DiamondWeFamily
