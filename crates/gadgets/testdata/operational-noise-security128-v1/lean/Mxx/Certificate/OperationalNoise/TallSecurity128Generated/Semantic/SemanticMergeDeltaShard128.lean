import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge25335
def owner : Owner := ⟨.program ⟨257⟩, ⟨20126⟩⟩
def mergeEvent : Nat := 25335
def frameStart : Nat := 25240
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩] } }
def leftRaw : List Term := Proof.Events098.exact25328RawTerms
def rightRaw : List Term := Proof.Events098.exact25285RawTerms
def group : MergeGroup := .operator 25328 25285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25328) (leftOrdinal := 0)
    (rightResult := 25285) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20123⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25335

namespace LeftMerge25343
def owner : Owner := ⟨.program ⟨257⟩, ⟨18520⟩⟩
def mergeEvent : Nat := 25343
def frameStart : Nat := 25240
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events098.exact25296RawTerms
def rightRaw : List Term := Proof.Events098.exact25339RawTerms
def group : MergeGroup := .operator 25296 25339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25296) (leftOrdinal := 0)
    (rightResult := 25339) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25343

namespace LeftMerge25360
def owner : Owner := ⟨.program ⟨257⟩, ⟨19065⟩⟩
def mergeEvent : Nat := 25360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12551⟩⟩, ⟨.program ⟨257⟩, ⟨18066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19657⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25357RawTerms
def group : MergeGroup := .relation 25359
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25359) (rhsResult := 25357)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19062⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25358 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19062⟩⟩]⟩) (none) 25357) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12551⟩⟩, ⟨.program ⟨257⟩, ⟨18066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19657⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12551⟩⟩, ⟨.program ⟨257⟩, ⟨18066⟩⟩], [⟨.program ⟨257⟩, ⟨19657⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25360

namespace LeftMerge25361
def owner : Owner := ⟨.program ⟨257⟩, ⟨19065⟩⟩
def mergeEvent : Nat := 25361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25357RawTerms
def group : MergeGroup := .relation 25359
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25359) (rhsResult := 25357)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19062⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25358 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19062⟩⟩]⟩) (none) 25357) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25361

namespace LeftMerge25362
def owner : Owner := ⟨.program ⟨257⟩, ⟨19065⟩⟩
def mergeEvent : Nat := 25362
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25357RawTerms
def group : MergeGroup := .relation 25359
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25359) (rhsResult := 25357)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19062⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25358 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19062⟩⟩]⟩) (none) 25357) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25362

namespace LeftMerge25363
def owner : Owner := ⟨.program ⟨257⟩, ⟨19065⟩⟩
def mergeEvent : Nat := 25363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25357RawTerms
def group : MergeGroup := .relation 25359
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25359) (rhsResult := 25357)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19062⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25358 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19062⟩⟩]⟩) (none) 25357) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25363

namespace LeftMerge25368
def owner : Owner := ⟨.program ⟨257⟩, ⟨20125⟩⟩
def mergeEvent : Nat := 25368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12551⟩⟩, ⟨.program ⟨257⟩, ⟨18066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19657⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25364RawTerms
def rightRaw : List Term := Proof.Events098.exact25178RawTerms
def group : MergeGroup := .operator 25364 25178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25364) (leftOrdinal := 2)
    (rightResult := 25178) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12551⟩⟩, ⟨.program ⟨257⟩, ⟨18066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19657⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12551⟩⟩, ⟨.program ⟨257⟩, ⟨18066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19657⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12551⟩⟩, ⟨.program ⟨257⟩, ⟨18066⟩⟩], [⟨.program ⟨257⟩, ⟨19657⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25368

namespace LeftMerge25369
def owner : Owner := ⟨.program ⟨257⟩, ⟨20125⟩⟩
def mergeEvent : Nat := 25369
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25364RawTerms
def rightRaw : List Term := Proof.Events098.exact25178RawTerms
def group : MergeGroup := .operator 25364 25178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25364) (leftOrdinal := 1)
    (rightResult := 25178) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20123⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25369

namespace LeftMerge25377
def owner : Owner := ⟨.program ⟨257⟩, ⟨20384⟩⟩
def mergeEvent : Nat := 25377
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25371RawTerms
def rightRaw : List Term := Proof.Events097.exact25075RawTerms
def group : MergeGroup := .operator 25371 25075
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25371) (leftOrdinal := 1)
    (rightResult := 25075) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20382⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25377

namespace LeftMerge25379
def owner : Owner := ⟨.program ⟨257⟩, ⟨20384⟩⟩
def mergeEvent : Nat := 25379
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19783⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact25072RawTerms
def group : MergeGroup := .relation 25378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25378) (rhsResult := 25072)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20382⟩⟩) ⟨19783⟩ 25072) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19783⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨19783⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25379

namespace LeftMerge25380
def owner : Owner := ⟨.program ⟨257⟩, ⟨20384⟩⟩
def mergeEvent : Nat := 25380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25371RawTerms
def rightRaw : List Term := Proof.Events097.exact25075RawTerms
def group : MergeGroup := .operator 25371 25075
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25371) (leftOrdinal := 0)
    (rightResult := 25075) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20382⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25380

namespace LeftMerge25394
def owner : Owner := ⟨.program ⟨257⟩, ⟨19285⟩⟩
def mergeEvent : Nat := 25394
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19282⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events099.exact25388RawTerms
def group : MergeGroup := .operator 17169 25388
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 25388) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25394

namespace LeftMerge25515
def owner : Owner := ⟨.program ⟨257⟩, ⟨20032⟩⟩
def mergeEvent : Nat := 25515
def frameStart : Nat := 25449
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25511RawTerms
def rightRaw : List Term := Proof.Events099.exact25509RawTerms
def group : MergeGroup := .operator 25511 25509
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25511) (leftOrdinal := 0)
    (rightResult := 25509) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25515

namespace LeftMerge25527
def owner : Owner := ⟨.program ⟨257⟩, ⟨20383⟩⟩
def mergeEvent : Nat := 25527
def frameStart : Nat := 25449
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25523RawTerms
def rightRaw : List Term := Proof.Events099.exact25500RawTerms
def group : MergeGroup := .operator 25523 25500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25523) (leftOrdinal := 1)
    (rightResult := 25500) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20382⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25527

namespace LeftMerge25529
def owner : Owner := ⟨.program ⟨257⟩, ⟨20383⟩⟩
def mergeEvent : Nat := 25529
def frameStart : Nat := 25449
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19783⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25497RawTerms
def group : MergeGroup := .relation 25528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25528) (rhsResult := 25497)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20382⟩⟩) ⟨19783⟩ 25497) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19783⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18518⟩⟩], [⟨.program ⟨257⟩, ⟨19783⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25529

namespace LeftMerge25530
def owner : Owner := ⟨.program ⟨257⟩, ⟨20383⟩⟩
def mergeEvent : Nat := 25530
def frameStart : Nat := 25449
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25523RawTerms
def rightRaw : List Term := Proof.Events099.exact25500RawTerms
def group : MergeGroup := .operator 25523 25500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25523) (leftOrdinal := 0)
    (rightResult := 25500) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20382⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20382⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25530

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
