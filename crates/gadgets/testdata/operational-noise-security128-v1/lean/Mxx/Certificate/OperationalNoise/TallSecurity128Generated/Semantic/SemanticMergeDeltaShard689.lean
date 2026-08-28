import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge115292
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115292
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40332⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 26)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40332⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115292

namespace LeftMerge115294
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115294
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40332⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115293) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115294

namespace LeftMerge115295
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115295
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 25)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115295

namespace LeftMerge115297
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115297
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115296
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115296) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115297

namespace LeftMerge115298
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115298
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34976⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 24)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34976⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115298

namespace LeftMerge115300
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115300
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34976⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115299
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115299) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115300

namespace LeftMerge115301
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115301
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 22)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115301

namespace LeftMerge115303
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115303
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115302
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115302) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115303

namespace LeftMerge115304
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115304
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 21)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115304

namespace LeftMerge115306
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115306
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26632⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115305) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115306

namespace LeftMerge115307
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115307
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 35)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115307

namespace LeftMerge115309
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115309
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115308) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115309

namespace LeftMerge115310
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115310
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 34)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115310

namespace LeftMerge115312
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115312
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115311) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115312

namespace LeftMerge115313
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115313
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩] } }
def leftRaw : List Term := Proof.Events450.exact115261RawTerms
def rightRaw : List Term := Proof.Events449.exact115102RawTerms
def group : MergeGroup := .operator 115261 115102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 115261) (leftOrdinal := 33)
    (rightResult := 115102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71267⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115313

namespace LeftMerge115315
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def mergeEvent : Nat := 115315
def frameStart : Nat := 114586
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60120⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }
def rhsRaw : List Term := Proof.Events449.exact115099RawTerms
def group : MergeGroup := .relation 115314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 115314) (rhsResult := 115099)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 115099) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68836⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge115315

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
