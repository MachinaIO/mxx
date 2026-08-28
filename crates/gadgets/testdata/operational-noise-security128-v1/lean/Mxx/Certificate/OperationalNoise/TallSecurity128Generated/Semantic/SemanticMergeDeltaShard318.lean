import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge55460
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55460
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 31)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55460

namespace LeftMerge55462
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55462
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55461
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55461) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55462

namespace LeftMerge55463
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55463
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 4)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55463

namespace LeftMerge55464
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55464
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 30)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55464

namespace LeftMerge55466
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55466
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55465) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55466

namespace LeftMerge55467
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55467
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 3)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55467

namespace LeftMerge55468
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 23)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55468

namespace LeftMerge55470
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55470
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55469) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55470

namespace LeftMerge55471
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 2)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55471

namespace LeftMerge55472
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 20)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55472

namespace LeftMerge55474
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55473
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55473) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55474

namespace LeftMerge55475
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 1)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55475

namespace LeftMerge55476
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55476
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 19)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55476

namespace LeftMerge55478
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55478
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46625RawTerms
def group : MergeGroup := .relation 55477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55477) (rhsResult := 46625)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55478

namespace LeftMerge55479
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55479
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 0)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55479

namespace LeftMerge55480
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def mergeEvent : Nat := 55480
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55405RawTerms
def rightRaw : List Term := Proof.Events182.exact46628RawTerms
def group : MergeGroup := .operator 55405 46628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55405) (leftOrdinal := 18)
    (rightResult := 46628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55480

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
