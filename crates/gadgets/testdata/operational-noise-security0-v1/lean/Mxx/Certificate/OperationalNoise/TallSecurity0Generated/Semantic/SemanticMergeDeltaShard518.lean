import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge84417
def owner : Owner := ⟨.program ⟨214⟩, ⟨14425⟩⟩
def mergeEvent : Nat := 84417
def frameStart : Nat := 84387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events329.exact84413RawTerms
def rightRaw : List Term := Proof.Events329.exact84410RawTerms
def group : MergeGroup := .operator 84413 84410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84413) (leftOrdinal := 0)
    (rightResult := 84410) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11553⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84417

namespace LeftMerge84447
def owner : Owner := ⟨.program ⟨214⟩, ⟨14533⟩⟩
def mergeEvent : Nat := 84447
def frameStart : Nat := 84387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events329.exact84443RawTerms
def rightRaw : List Term := Proof.Events329.exact84441RawTerms
def group : MergeGroup := .operator 84443 84441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84443) (leftOrdinal := 0)
    (rightResult := 84441) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84447

namespace LeftMerge84468
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def mergeEvent : Nat := 84468
def frameStart : Nat := 84387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events329.exact84464RawTerms
def rightRaw : List Term := Proof.Events329.exact84461RawTerms
def group : MergeGroup := .operator 84464 84461
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84464) (leftOrdinal := 0)
    (rightResult := 84461) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84468

namespace LeftMerge84477
def owner : Owner := ⟨.program ⟨214⟩, ⟨26146⟩⟩
def mergeEvent : Nat := 84477
def frameStart : Nat := 84387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩] } }
def leftRaw : List Term := Proof.Events329.exact84473RawTerms
def rightRaw : List Term := Proof.Events329.exact84432RawTerms
def group : MergeGroup := .operator 84473 84432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84473) (leftOrdinal := 0)
    (rightResult := 84432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26143⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84477

namespace LeftMerge84478
def owner : Owner := ⟨.program ⟨214⟩, ⟨26146⟩⟩
def mergeEvent : Nat := 84478
def frameStart : Nat := 84387
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩] } }
def leftRaw : List Term := Proof.Events329.exact84473RawTerms
def rightRaw : List Term := Proof.Events329.exact84432RawTerms
def group : MergeGroup := .operator 84473 84432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84473) (leftOrdinal := 1)
    (rightResult := 84432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26143⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84478

namespace LeftMerge84480
def owner : Owner := ⟨.program ⟨214⟩, ⟨26146⟩⟩
def mergeEvent : Nat := 84480
def frameStart : Nat := 84387
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23626⟩⟩] } }
def rhsRaw : List Term := Proof.Events329.exact84429RawTerms
def group : MergeGroup := .relation 84479
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84479) (rhsResult := 84429)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26143⟩⟩) ⟨23626⟩ 84429) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨23626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84480

namespace LeftMerge84488
def owner : Owner := ⟨.program ⟨214⟩, ⟨16061⟩⟩
def mergeEvent : Nat := 84488
def frameStart : Nat := 84387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16059⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events329.exact84443RawTerms
def rightRaw : List Term := Proof.Events330.exact84484RawTerms
def group : MergeGroup := .operator 84443 84484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84443) (leftOrdinal := 0)
    (rightResult := 84484) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16059⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84488

namespace LeftMerge84505
def owner : Owner := ⟨.program ⟨214⟩, ⟨19603⟩⟩
def mergeEvent : Nat := 84505
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }
def rhsRaw : List Term := Proof.Events330.exact84502RawTerms
def group : MergeGroup := .relation 84504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84504) (rhsResult := 84502)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84503 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩) (none) 84502) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84505

namespace LeftMerge84506
def owner : Owner := ⟨.program ⟨214⟩, ⟨19603⟩⟩
def mergeEvent : Nat := 84506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩] } }
def rhsRaw : List Term := Proof.Events330.exact84502RawTerms
def group : MergeGroup := .relation 84504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84504) (rhsResult := 84502)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84503 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩) (none) 84502) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84506

namespace LeftMerge84507
def owner : Owner := ⟨.program ⟨214⟩, ⟨19603⟩⟩
def mergeEvent : Nat := 84507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23626⟩⟩] } }
def rhsRaw : List Term := Proof.Events330.exact84502RawTerms
def group : MergeGroup := .relation 84504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84504) (rhsResult := 84502)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84503 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩) (none) 84502) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨23626⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84507

namespace LeftMerge84508
def owner : Owner := ⟨.program ⟨214⟩, ⟨19603⟩⟩
def mergeEvent : Nat := 84508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events330.exact84502RawTerms
def group : MergeGroup := .relation 84504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84504) (rhsResult := 84502)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84503 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩) (none) 84502) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16059⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84508

namespace LeftMerge84513
def owner : Owner := ⟨.program ⟨214⟩, ⟨26145⟩⟩
def mergeEvent : Nat := 84513
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23626⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84509RawTerms
def rightRaw : List Term := Proof.Events329.exact84325RawTerms
def group : MergeGroup := .operator 84509 84325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84509) (leftOrdinal := 2)
    (rightResult := 84325) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23626⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23626⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨23626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84513

namespace LeftMerge84514
def owner : Owner := ⟨.program ⟨214⟩, ⟨26145⟩⟩
def mergeEvent : Nat := 84514
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84509RawTerms
def rightRaw : List Term := Proof.Events329.exact84325RawTerms
def group : MergeGroup := .operator 84509 84325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84509) (leftOrdinal := 1)
    (rightResult := 84325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84514

namespace LeftMerge84522
def owner : Owner := ⟨.program ⟨214⟩, ⟨28085⟩⟩
def mergeEvent : Nat := 84522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28083⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84516RawTerms
def rightRaw : List Term := Proof.Events329.exact84241RawTerms
def group : MergeGroup := .operator 84516 84241
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84516) (leftOrdinal := 0)
    (rightResult := 84241) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28083⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28083⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84522

namespace LeftMerge84523
def owner : Owner := ⟨.program ⟨214⟩, ⟨28085⟩⟩
def mergeEvent : Nat := 84523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28083⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84516RawTerms
def rightRaw : List Term := Proof.Events329.exact84241RawTerms
def group : MergeGroup := .operator 84516 84241
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84516) (leftOrdinal := 1)
    (rightResult := 84241) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28083⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84523

namespace LeftMerge84525
def owner : Owner := ⟨.program ⟨214⟩, ⟨28085⟩⟩
def mergeEvent : Nat := 84525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24225⟩⟩] } }
def rhsRaw : List Term := Proof.Events329.exact84238RawTerms
def group : MergeGroup := .relation 84524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84524) (rhsResult := 84238)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28083⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28083⟩⟩) ⟨24225⟩ 84238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24225⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨24225⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84525

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
