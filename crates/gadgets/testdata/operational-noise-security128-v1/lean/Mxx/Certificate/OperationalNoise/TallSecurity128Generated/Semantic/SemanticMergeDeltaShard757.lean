import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge125161
def owner : Owner := ⟨.program ⟨257⟩, ⟨58436⟩⟩
def mergeEvent : Nat := 125161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩] } }
def leftRaw : List Term := Proof.Events488.exact125155RawTerms
def rightRaw : List Term := Proof.Events488.exact125091RawTerms
def group : MergeGroup := .operator 125155 125091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125155) (leftOrdinal := 1)
    (rightResult := 125091) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58435⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125161

namespace LeftMerge125163
def owner : Owner := ⟨.program ⟨257⟩, ⟨58436⟩⟩
def mergeEvent : Nat := 125163
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } }
def rhsRaw : List Term := Proof.Events488.exact125088RawTerms
def group : MergeGroup := .relation 125162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125162) (rhsResult := 125088)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58435⟩⟩) ⟨57945⟩ 125088) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨57945⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125163

namespace LeftMerge125164
def owner : Owner := ⟨.program ⟨257⟩, ⟨58436⟩⟩
def mergeEvent : Nat := 125164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩] } }
def leftRaw : List Term := Proof.Events488.exact125155RawTerms
def rightRaw : List Term := Proof.Events488.exact125091RawTerms
def group : MergeGroup := .operator 125155 125091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125155) (leftOrdinal := 0)
    (rightResult := 125091) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58435⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125164

namespace LeftMerge125178
def owner : Owner := ⟨.program ⟨257⟩, ⟨57372⟩⟩
def mergeEvent : Nat := 125178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events488.exact125172RawTerms
def group : MergeGroup := .operator 119870 125172
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 125172) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57369⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125178

namespace LeftMerge125257
def owner : Owner := ⟨.program ⟨257⟩, ⟨56398⟩⟩
def mergeEvent : Nat := 125257
def frameStart : Nat := 125227
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events489.exact125253RawTerms
def rightRaw : List Term := Proof.Events489.exact125250RawTerms
def group : MergeGroup := .operator 125253 125250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125253) (leftOrdinal := 0)
    (rightResult := 125250) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24962⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125257

namespace LeftMerge125287
def owner : Owner := ⟨.program ⟨257⟩, ⟨58232⟩⟩
def mergeEvent : Nat := 125287
def frameStart : Nat := 125227
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events489.exact125283RawTerms
def rightRaw : List Term := Proof.Events489.exact125281RawTerms
def group : MergeGroup := .operator 125283 125281
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125283) (leftOrdinal := 0)
    (rightResult := 125281) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125287

namespace LeftMerge125310
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 125310
def frameStart : Nat := 125227
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events489.exact125306RawTerms
def rightRaw : List Term := Proof.Events489.exact125303RawTerms
def group : MergeGroup := .operator 125306 125303
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125306) (leftOrdinal := 0)
    (rightResult := 125303) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125310

namespace LeftMerge125319
def owner : Owner := ⟨.program ⟨257⟩, ⟨58438⟩⟩
def mergeEvent : Nat := 125319
def frameStart : Nat := 125227
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩] } }
def leftRaw : List Term := Proof.Events489.exact125315RawTerms
def rightRaw : List Term := Proof.Events489.exact125272RawTerms
def group : MergeGroup := .operator 125315 125272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125315) (leftOrdinal := 0)
    (rightResult := 125272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58435⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125319

namespace LeftMerge125320
def owner : Owner := ⟨.program ⟨257⟩, ⟨58438⟩⟩
def mergeEvent : Nat := 125320
def frameStart : Nat := 125227
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩] } }
def leftRaw : List Term := Proof.Events489.exact125315RawTerms
def rightRaw : List Term := Proof.Events489.exact125272RawTerms
def group : MergeGroup := .operator 125315 125272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125315) (leftOrdinal := 1)
    (rightResult := 125272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58435⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125320

namespace LeftMerge125322
def owner : Owner := ⟨.program ⟨257⟩, ⟨58438⟩⟩
def mergeEvent : Nat := 125322
def frameStart : Nat := 125227
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } }
def rhsRaw : List Term := Proof.Events489.exact125269RawTerms
def group : MergeGroup := .relation 125321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125321) (rhsResult := 125269)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58435⟩⟩) ⟨57945⟩ 125269) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨57945⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125322

namespace LeftMerge125330
def owner : Owner := ⟨.program ⟨257⟩, ⟨56818⟩⟩
def mergeEvent : Nat := 125330
def frameStart : Nat := 125227
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events489.exact125283RawTerms
def rightRaw : List Term := Proof.Events489.exact125326RawTerms
def group : MergeGroup := .operator 125283 125326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125283) (leftOrdinal := 0)
    (rightResult := 125326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56816⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56816⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125330

namespace LeftMerge125347
def owner : Owner := ⟨.program ⟨257⟩, ⟨57372⟩⟩
def mergeEvent : Nat := 125347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events489.exact125344RawTerms
def group : MergeGroup := .relation 125346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125346) (rhsResult := 125344)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 125345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩) (none) 125344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125347

namespace LeftMerge125348
def owner : Owner := ⟨.program ⟨257⟩, ⟨57372⟩⟩
def mergeEvent : Nat := 125348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩] } }
def rhsRaw : List Term := Proof.Events489.exact125344RawTerms
def group : MergeGroup := .relation 125346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125346) (rhsResult := 125344)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 125345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩) (none) 125344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58435⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125348

namespace LeftMerge125349
def owner : Owner := ⟨.program ⟨257⟩, ⟨57372⟩⟩
def mergeEvent : Nat := 125349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } }
def rhsRaw : List Term := Proof.Events489.exact125344RawTerms
def group : MergeGroup := .relation 125346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125346) (rhsResult := 125344)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 125345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩) (none) 125344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨57945⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125349

namespace LeftMerge125350
def owner : Owner := ⟨.program ⟨257⟩, ⟨57372⟩⟩
def mergeEvent : Nat := 125350
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨56816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events489.exact125344RawTerms
def group : MergeGroup := .relation 125346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125346) (rhsResult := 125344)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 125345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57369⟩⟩]⟩) (none) 125344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨56816⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125350

namespace LeftMerge125355
def owner : Owner := ⟨.program ⟨257⟩, ⟨58437⟩⟩
def mergeEvent : Nat := 125355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } }
def leftRaw : List Term := Proof.Events489.exact125351RawTerms
def rightRaw : List Term := Proof.Events488.exact125165RawTerms
def group : MergeGroup := .operator 125351 125165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125351) (leftOrdinal := 2)
    (rightResult := 125165) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57945⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24962⟩⟩, ⟨.program ⟨257⟩, ⟨56397⟩⟩], [⟨.program ⟨257⟩, ⟨57945⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125355

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
