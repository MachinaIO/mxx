import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge213349
def owner : Owner := ⟨.program ⟨257⟩, ⟨53528⟩⟩
def mergeEvent : Nat := 213349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events833.exact213343RawTerms
def rightRaw : List Term := Proof.Events039.exact10097RawTerms
def group : MergeGroup := .operator 213343 10097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213343) (leftOrdinal := 1)
    (rightResult := 10097) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213349

namespace LeftMerge213350
def owner : Owner := ⟨.program ⟨257⟩, ⟨53528⟩⟩
def mergeEvent : Nat := 213350
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }
def leftRaw : List Term := Proof.Events833.exact213343RawTerms
def rightRaw : List Term := Proof.Events039.exact10097RawTerms
def group : MergeGroup := .operator 213343 10097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213343) (leftOrdinal := 0)
    (rightResult := 10097) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213350

namespace LeftMerge213355
def owner : Owner := ⟨.program ⟨257⟩, ⟨53529⟩⟩
def mergeEvent : Nat := 213355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10097RawTerms
def rightRaw : List Term := Proof.Events810.exact207528RawTerms
def group : MergeGroup := .operator 10097 207528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10097) (leftOrdinal := 0)
    (rightResult := 207528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213355

namespace LeftMerge213360
def owner : Owner := ⟨.program ⟨257⟩, ⟨8595⟩⟩
def mergeEvent : Nat := 213360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }
def leftRaw : List Term := Proof.Events810.exact207398RawTerms
def rightRaw : List Term := Proof.Events090.exact23133RawTerms
def group : MergeGroup := .operator 207398 23133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207398) (leftOrdinal := 0)
    (rightResult := 23133) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213360

namespace LeftMerge213377
def owner : Owner := ⟨.program ⟨257⟩, ⟨53532⟩⟩
def mergeEvent : Nat := 213377
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events833.exact213371RawTerms
def rightRaw : List Term := Proof.Events090.exact23122RawTerms
def group : MergeGroup := .operator 213371 23122
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213371) (leftOrdinal := 1)
    (rightResult := 23122) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213377

namespace LeftMerge213379
def owner : Owner := ⟨.program ⟨257⟩, ⟨53532⟩⟩
def mergeEvent : Nat := 213379
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }
def rhsRaw : List Term := Proof.Events090.exact23092RawTerms
def group : MergeGroup := .relation 213378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213378) (rhsResult := 23092)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9529⟩⟩) ⟨7272⟩ 23092) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213379

namespace LeftMerge213380
def owner : Owner := ⟨.program ⟨257⟩, ⟨53532⟩⟩
def mergeEvent : Nat := 213380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events833.exact213371RawTerms
def rightRaw : List Term := Proof.Events090.exact23122RawTerms
def group : MergeGroup := .operator 213371 23122
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213371) (leftOrdinal := 0)
    (rightResult := 23122) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213380

namespace LeftMerge213385
def owner : Owner := ⟨.program ⟨257⟩, ⟨53533⟩⟩
def mergeEvent : Nat := 213385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }
def leftRaw : List Term := Proof.Events833.exact213381RawTerms
def rightRaw : List Term := Proof.Events833.exact213351RawTerms
def group : MergeGroup := .operator 213381 213351
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213381) (leftOrdinal := 1)
    (rightResult := 213351) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213385

namespace LeftMerge213393
def owner : Owner := ⟨.program ⟨257⟩, ⟨55500⟩⟩
def mergeEvent : Nat := 213393
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } }
def leftRaw : List Term := Proof.Events833.exact213387RawTerms
def rightRaw : List Term := Proof.Events833.exact213323RawTerms
def group : MergeGroup := .operator 213387 213323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213387) (leftOrdinal := 1)
    (rightResult := 213323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213393

namespace LeftMerge213395
def owner : Owner := ⟨.program ⟨257⟩, ⟨55500⟩⟩
def mergeEvent : Nat := 213395
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } }
def rhsRaw : List Term := Proof.Events833.exact213320RawTerms
def group : MergeGroup := .relation 213394
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213394) (rhsResult := 213320)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55499⟩⟩) ⟨54989⟩ 213320) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨54989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213395

namespace LeftMerge213396
def owner : Owner := ⟨.program ⟨257⟩, ⟨55500⟩⟩
def mergeEvent : Nat := 213396
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } }
def leftRaw : List Term := Proof.Events833.exact213387RawTerms
def rightRaw : List Term := Proof.Events833.exact213323RawTerms
def group : MergeGroup := .operator 213387 213323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213387) (leftOrdinal := 0)
    (rightResult := 213323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213396

namespace LeftMerge213410
def owner : Owner := ⟨.program ⟨257⟩, ⟨54432⟩⟩
def mergeEvent : Nat := 213410
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events833.exact213404RawTerms
def group : MergeGroup := .operator 207620 213404
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 213404) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54429⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213410

namespace LeftMerge213489
def owner : Owner := ⟨.program ⟨257⟩, ⟨53526⟩⟩
def mergeEvent : Nat := 213489
def frameStart : Nat := 213459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events833.exact213485RawTerms
def rightRaw : List Term := Proof.Events833.exact213482RawTerms
def group : MergeGroup := .operator 213485 213482
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213485) (leftOrdinal := 0)
    (rightResult := 213482) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213489

namespace LeftMerge213519
def owner : Owner := ⟨.program ⟨257⟩, ⟨55268⟩⟩
def mergeEvent : Nat := 213519
def frameStart : Nat := 213459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213515RawTerms
def rightRaw : List Term := Proof.Events834.exact213513RawTerms
def group : MergeGroup := .operator 213515 213513
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213515) (leftOrdinal := 0)
    (rightResult := 213513) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213519

namespace LeftMerge213542
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 213542
def frameStart : Nat := 213459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213538RawTerms
def rightRaw : List Term := Proof.Events834.exact213535RawTerms
def group : MergeGroup := .operator 213538 213535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213538) (leftOrdinal := 0)
    (rightResult := 213535) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213542

namespace LeftMerge213551
def owner : Owner := ⟨.program ⟨257⟩, ⟨55502⟩⟩
def mergeEvent : Nat := 213551
def frameStart : Nat := 213459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213547RawTerms
def rightRaw : List Term := Proof.Events834.exact213504RawTerms
def group : MergeGroup := .operator 213547 213504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213547) (leftOrdinal := 0)
    (rightResult := 213504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55499⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213551

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
